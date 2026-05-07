import os
import argparse
import sys
import base64
import hashlib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import flwr as fl
from cryptography.hazmat.primitives.asymmetric import x25519

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as proj_cfg
from src.model.face_model import get_model, get_parameters, set_parameters


def hash_parameters(ndarrays) -> str:
    """Create a stable SHA-256 hash for parameter payload metadata/signing."""
    hasher = hashlib.sha256()
    for arr in ndarrays:
        hasher.update(arr.tobytes())
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
    return hasher.hexdigest()


class FaceDataset(Dataset):
    """
    Reads .pt face tensor files from a client folder.

    Since each client has only ONE person's photos, we assign:
        label = 1 for all real photos (positive samples)
    We also create synthetic "negative" pairs on the fly during training
    using CosineEmbeddingLoss which needs pairs, not class indices.

    But for simplicity here: we just use the tensor + label=0.
    The loss function (MSELoss on embeddings) doesn't need multi-class labels.
    """

    def __init__(self, client_dir):
        self.files = []
        for root, _, files in os.walk(client_dir):
            for file in files:
                if file.endswith('.pt'):
                    self.files.append(os.path.join(root, file))

        if len(self.files) == 0:
            print(f"[WARNING] No .pt files found in {client_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tensor = torch.load(self.files[idx])   # shape [3, 160, 160]
        return tensor


# =============================================================================
# Training with embedding-based loss
# =============================================================================

def train_local(model, loader, optimizer, epochs, dp_mode="none"):
    """
    Train the model locally using embedding consistency loss.

    Supports different DP modes:
        - "opacus": No change here (model is already wrapped)
        - "manual_sgd": Clips gradients and adds noise manually
        - "embedding": Adds noise to the 512-dim embedding vectors
        - "none" / "client": Plain training
    """
    from src.augmentation.augment import FaceAugmentation
    from src.privacy.dp_training import apply_manual_dp_sgd, apply_embedding_noise

    augment = FaceAugmentation(use_affine=True, use_blur=False)
    model.train()
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            images = batch

            # PyTorch BatchNorm crashes on batch size 1 — duplicate if needed
            if len(images) == 1:
                images = torch.cat([images, images], dim=0)

            optimizer.zero_grad()

            emb1 = model(images)
            images_aug = torch.stack([augment(img) for img in images])
            emb2 = model(images_aug)

            # MODE: Embedding DP (Local DP)
            if dp_mode == "embedding":
                emb1 = apply_embedding_noise(emb1, proj_cfg.NOISE_MULTIPLIER * 0.1)
                emb2 = apply_embedding_noise(emb2, proj_cfg.NOISE_MULTIPLIER * 0.1)

            loss = criterion(emb1, emb2)
            loss.backward()

            # MODE: Manual SGD DP
            if dp_mode == "manual_sgd":
                apply_manual_dp_sgd(
                    model, 
                    noise_multiplier=proj_cfg.NOISE_MULTIPLIER, 
                    max_grad_norm=proj_cfg.MAX_GRAD_NORM
                )

            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / max(len(loader), 1)
        print(f"    Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")


def evaluate_local(model, loader):
    """
    Evaluate by measuring average embedding consistency.
    Lower MSE between original and augmented views = better model.
    We convert this to a pseudo-accuracy: acc = 1 - normalised_loss
    so the server gets a number between 0 and 1 (higher is better).
    """
    from src.augmentation.augment import FaceAugmentation

    augment = FaceAugmentation(use_affine=False, use_blur=False)
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch
            emb1 = model(images)
            images_aug = torch.stack([augment(img) for img in images])
            emb2 = model(images_aug)
            total_loss += criterion(emb1, emb2).item()

    avg_loss = total_loss / max(len(loader), 1)
    pseudo_accuracy = max(0.0, 1.0 - avg_loss)
    return avg_loss, pseudo_accuracy


# =============================================================================
# Flower client
# =============================================================================

class FaceClient(fl.client.NumPyClient):
    """
    FL client. Flower calls get_parameters, fit, evaluate automatically.

    fit() is where Version A and Version B split:
        use_dp=False → plain training   (Version A)
        use_dp=True  → Opacus wraps the optimizer with DP  (Version B)

    Security layers active in fit():
        1. DP training (Version B only) — noise added to gradients before update
        2. Payload encryption probe    — proves AES-256-GCM is working each round
        3. Ed25519 signing             — update is signed before sending to server
        4. Audit logging               — every round event is logged with hash chain
    """

    def __init__(self, client_id, use_dp):
        if isinstance(client_id, str) and not client_id.isdigit():
            self.client_id = f"client_{client_id}"
        else:
            self.client_id = f"client_{int(client_id):02d}"
        self.client_dir = os.path.join(proj_cfg.CLIENTS_DIR, self.client_id)
        self.use_dp = use_dp
        self.local_round = 0
        self.last_epsilon = 0.0
        self.crypto_debug_logs = getattr(proj_cfg, "CRYPTO_DEBUG_LOGS", False)
        self.crypto_debug_show_full_keys = getattr(proj_cfg, "CRYPTO_DEBUG_SHOW_FULL_KEYS", False)
        self.payload_encryptor = None
        self.payload_decryptor = None
        self.server_private_key_hex = None
        self.server_public_key_hex = None

        self.model = get_model(mode="train")

        from src.privacy.privacy_accounting import PrivacyAccountant
        from src.privacy.signing import generate_client_keypair, load_private_key
        from src.privacy.audit_logging import get_audit_log

        self.accountant = PrivacyAccountant(delta=proj_cfg.DELTA)

        # FIX: Audit log is now wired into the client — every round gets logged
        log_path = os.path.join(proj_cfg.METRICS_DIR, f"audit_{self.client_id}.log")
        self.audit_log = get_audit_log(log_path)

        os.makedirs(proj_cfg.KEYS_DIR, exist_ok=True)

        self.private_key_path = os.path.join(proj_cfg.KEYS_DIR, f"{self.client_id}_private.pem")
        self.public_key_path = os.path.join(proj_cfg.KEYS_DIR, f"{self.client_id}_public.pem")
        if not os.path.exists(self.private_key_path) or not os.path.exists(self.public_key_path):
            generate_client_keypair(self.client_id, output_dir=proj_cfg.KEYS_DIR)
        self.private_key = load_private_key(self.private_key_path)

        # Build dataset
        dataset = FaceDataset(self.client_dir)

        if len(dataset) == 0:
            print(f"[{self.client_id}] ERROR: No photos found in {self.client_dir}. Terminating.")
            sys.exit(0)
        else:
            self.num_samples = len(dataset)

        self.train_loader = DataLoader(
            dataset, batch_size=8, shuffle=True,
            collate_fn=lambda x: torch.stack(x)
        )
        self.eval_loader = DataLoader(
            dataset, batch_size=8, shuffle=False,
            collate_fn=lambda x: torch.stack(x)
        )

        # Log client startup to audit trail
        self.audit_log.log_event(
            event_type="client_initialized",
            details={
                "client_id": self.client_id,
                "num_samples": self.num_samples,
                "dp_enabled": self.use_dp,
            },
            severity="INFO"
        )

        print(f"\n[{self.client_id}] Initialization successful! Ready to connect to Server.")
        print(f"  Photos: {self.num_samples}")
        print(f"  DP:     {'ON  → Version B' if use_dp else 'OFF → Version A'}")
        if self.crypto_debug_logs:
            print(f"  Crypto verification logs: ON")

    def _show_key(self, key_hex: str) -> str:
        if self.crypto_debug_show_full_keys:
            return key_hex
        return f"{key_hex[:16]}...{key_hex[-16:]}"

    def _init_payload_crypto_probe(self):
        """
        Initialize a local X25519 server keypair used only for runtime encryption verification.

        NOTE: This is a PROBE — it proves the crypto module works correctly each round.
        The actual model weights sent to the real server are NOT yet encrypted end-to-end
        through the Flower transport layer. Full production wiring is the next integration step.
        """
        if self.payload_encryptor is not None and self.payload_decryptor is not None:
            return

        from src.privacy.payload_encryption import PayloadEncryptor, PayloadDecryptor

        server_private = x25519.X25519PrivateKey.generate()
        server_public = server_private.public_key()

        server_private_bytes = server_private.private_bytes_raw()
        server_public_bytes = server_public.public_bytes_raw()

        self.server_private_key_hex = server_private_bytes.hex()
        self.server_public_key_hex = server_public_bytes.hex()

        self.payload_encryptor = PayloadEncryptor(server_public_bytes)
        self.payload_decryptor = PayloadDecryptor(server_private_bytes)

        if self.crypto_debug_logs:
            print(f"[{self.client_id}] [CRYPTO-VERIFY] Payload crypto probe initialized")
            print(f"[{self.client_id}] [CRYPTO-VERIFY] X25519 server_public_key={self._show_key(self.server_public_key_hex)}")
            print(f"[{self.client_id}] [CRYPTO-VERIFY] X25519 server_private_key={self._show_key(self.server_private_key_hex)}")

    def _run_payload_encryption_probe(self, model_hash: str):
        """
        Encrypt/decrypt a per-round payload and verify correctness.

        This probe runs every round and:
        - Encrypts the model hash using AES-256-GCM + X25519
        - Immediately decrypts it and checks the result matches
        - Logs success/failure to the audit trail

        If this probe fails, it means the crypto module has broken — we catch and log it
        instead of crashing the whole training run.
        """
        self._init_payload_crypto_probe()

        probe_payload = model_hash.encode("utf-8")
        try:
            encrypted_data = self.payload_encryptor.encrypt(
                plaintext=probe_payload,
                round_id=self.local_round,
                client_id=self.client_id,
                protocol_version=proj_cfg.PROTOCOL_VERSION,
            )

            recovered = self.payload_decryptor.decrypt(encrypted_data)
            ok = recovered == probe_payload
        except Exception as e:
            # FIX: Log crypto failures to audit trail instead of silently passing
            self.audit_log.log_event(
                event_type="crypto_probe_failed",
                details={"round": self.local_round, "error": str(e)},
                severity="ERROR"
            )
            print(f"[{self.client_id}] [CRYPTO-VERIFY] ERROR: Probe failed: {e}")
            return {"crypto_probe_ok": 0}

        encrypted_payload_hex = encrypted_data["encrypted_payload"]
        nonce_hex = encrypted_payload_hex[:24]
        encrypted_len = len(bytes.fromhex(encrypted_payload_hex))
        ephemeral_public_hex = encrypted_data["ephemeral_public_key"]

        # FIX: Log crypto probe result to audit trail every round
        self.audit_log.log_event(
            event_type="crypto_probe",
            details={
                "round": self.local_round,
                "probe_ok": ok,
                "algorithm": encrypted_data.get("algorithm", "AES-256-GCM"),
                "key_agreement": encrypted_data.get("key_agreement", "X25519"),
            },
            severity="INFO" if ok else "ERROR"
        )

        if self.crypto_debug_logs:
            print(f"[{self.client_id}] [CRYPTO-VERIFY] Round={self.local_round} AAD=(round_id={self.local_round}, client_id={self.client_id}, protocol_version={proj_cfg.PROTOCOL_VERSION})")
            print(f"[{self.client_id}] [CRYPTO-VERIFY] AES-GCM nonce={nonce_hex}")
            print(f"[{self.client_id}] [CRYPTO-VERIFY] Ephemeral X25519 public_key={self._show_key(ephemeral_public_hex)}")
            print(f"[{self.client_id}] [CRYPTO-VERIFY] Ciphertext+tag bytes={encrypted_len}")
            print(f"[{self.client_id}] [CRYPTO-VERIFY] Encrypt->Decrypt match={ok}")

        return {
            "crypto_probe_ok": int(ok),
            "crypto_nonce_hex": nonce_hex,
            "crypto_ephemeral_public_key": ephemeral_public_hex,
            "crypto_payload_bytes": int(encrypted_len),
            "crypto_algorithm": encrypted_data.get("algorithm", "AES-256-GCM"),
            "crypto_key_agreement": encrypted_data.get("key_agreement", "X25519"),
        }

    def get_parameters(self, config):
        """Return current model weights as numpy arrays for Flower."""
        print(f"[{self.client_id}] Server requested parameters...")
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """
        Core of FL: receive global weights, train locally, return updated weights.

        Steps:
            1. Load global weights sent by server into local model
            2. Create Adam optimizer
            3a. Version A: train normally
            3b. Version B: wrap with Opacus (DP) then train
            4. Sign the update manifest with Ed25519
            5. Run encryption probe to verify crypto is working
            6. Log everything to tamper-evident audit trail
            7. Return updated weights + metrics to server
        """

        # Step 1 — load global weights
        set_parameters(self.model, parameters)
        self.local_round += 1

        # Log round start to audit trail
        self.audit_log.log_fl_round(
            round_number=self.local_round,
            num_clients=1,
            num_samples=self.num_samples,
        )

        # Step 2 — create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=proj_cfg.LEARNING_RATE)

        # Step 3 — train (this is where A and B split)
        if self.use_dp:
            # VERSION B: Uses Differential Privacy Module
            from src.privacy.dp_training import make_private_with_dp, PrivacyBudgetExceeded, apply_client_dp_noise, PrivacyMonitor, _freeze_backbone_for_dp

            dp_mode = getattr(proj_cfg, "DP_MODE", "opacus")

            # Log that DP is enabled this round
            self.audit_log.log_event(
                event_type="dp_mode_active",
                details={"mode": dp_mode, "noise": proj_cfg.NOISE_MULTIPLIER}
            )

            if dp_mode == "opacus":
                model_dp, optimizer_dp, loader_dp, privacy_engine, privacy_monitor = make_private_with_dp(
                    model=self.model,
                    optimizer=optimizer,
                    train_loader=self.train_loader,
                    noise_multiplier=proj_cfg.NOISE_MULTIPLIER,
                    max_grad_norm=proj_cfg.MAX_GRAD_NORM,
                    delta=proj_cfg.DELTA,
                    epsilon_max=proj_cfg.EPSILON_MAX,
                    random_seed=proj_cfg.DP_RANDOM_SEED,
                    client_id=self.client_id,
                    freeze_backbone=False,
                )
                print(f"[{self.client_id}] Training with OPACUS DP-SGD...")
                try:
                    train_local(model_dp, loader_dp, optimizer_dp, proj_cfg.LOCAL_EPOCHS, dp_mode="opacus")
                    epsilon = privacy_monitor.check_and_log(privacy_engine)
                except PrivacyBudgetExceeded as e:
                    print(f"[{self.client_id}] Privacy budget exceeded: {e}")
                    epsilon = privacy_engine.get_epsilon(delta=proj_cfg.DELTA)
                self.last_epsilon = float(epsilon)

            elif dp_mode in ["manual_sgd", "embedding", "client"]:
                print(f"[{self.client_id}] Training with {dp_mode.upper()} DP mode...")
                
                # For manual/embedding modes, we still want to freeze backbone to keep it fair/efficient
                _freeze_backbone_for_dp(self.model)
                
                train_local(self.model, self.train_loader, optimizer, proj_cfg.LOCAL_EPOCHS, dp_mode=dp_mode)
                
                if dp_mode == "client":
                    apply_client_dp_noise(self.model, proj_cfg.NOISE_MULTIPLIER * 0.01)

                # Accounting for manual modes (simplified)
                from src.privacy.dp_training import get_manual_epsilon
                steps = proj_cfg.LOCAL_EPOCHS * len(self.train_loader)
                epsilon = get_manual_epsilon(
                    steps=steps,
                    batch_size=8,
                    total_samples=self.num_samples,
                    noise_multiplier=proj_cfg.NOISE_MULTIPLIER if dp_mode == "manual_sgd" else 0.0,
                    delta=proj_cfg.DELTA
                )
                self.last_epsilon = float(epsilon)
            else:
                # Fallback to plain training if mode unknown
                train_local(self.model, self.train_loader, optimizer, proj_cfg.LOCAL_EPOCHS)
                self.last_epsilon = 0.0

            # FIX: Log epsilon consumption to audit trail after every round
            self.audit_log.log_epsilon_update(
                round_number=self.local_round,
                cumulative_epsilon=self.last_epsilon,
                delta=proj_cfg.DELTA,
            )

            print(f"[{self.client_id}] Privacy budget spent: epsilon = {epsilon:.2f}")
        else:
            # VERSION A: plain training, no DP protection
            print(f"[{self.client_id}] Training WITHOUT DP...")
            train_local(self.model, self.train_loader, optimizer, proj_cfg.LOCAL_EPOCHS)
            self.last_epsilon = 0.0

        # Step 4 — sign the update before sending
        from src.privacy.signing import sign_update

        updated_params = get_parameters(self.model)
        model_hash = hash_parameters(updated_params)

        signed_manifest = {
            "round_id": self.local_round,
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "model_hash": model_hash,
            "protocol_version": proj_cfg.PROTOCOL_VERSION,
        }
        signature = sign_update(self.private_key, signed_manifest)

        # Step 5 — run encryption probe
        crypto_metrics = self._run_payload_encryption_probe(model_hash)

        # Step 6 — log the signed update to audit trail
        self.audit_log.log_client_update(
            round_number=self.local_round,
            client_id=self.client_id,
            num_samples=self.num_samples,
            signature_valid=True,  # We just signed it ourselves so it's valid
        )

        # Step 7 — return to server
        metrics = {
            "round_id": self.local_round,
            "client_id": self.client_id,
            "protocol_version": proj_cfg.PROTOCOL_VERSION,
            "model_hash": model_hash,
            "signature_b64": base64.b64encode(signature).decode("utf-8"),
            "epsilon": float(self.last_epsilon),
        }
        metrics.update(crypto_metrics)

        return updated_params, self.num_samples, metrics

    def evaluate(self, parameters, config):
        """
        Measure how good the current global model is on local data.
        Returns real accuracy (not hardcoded 1.0).
        """
        set_parameters(self.model, parameters)
        loss, accuracy = evaluate_local(self.model, self.eval_loader)

        self.accountant.log_round(
            round_number=self.local_round,
            epsilon=float(self.last_epsilon),
            accuracy=float(accuracy),
            loss=float(loss),
        )

        # FIX: Log evaluation result to audit trail
        self.audit_log.log_event(
            event_type="evaluation",
            details={
                "round": self.local_round,
                "loss": float(loss),
                "accuracy": float(accuracy),
                "epsilon": float(self.last_epsilon),
            },
            severity="INFO"
        )

        if self.local_round == proj_cfg.NUM_ROUNDS:
            report_path = os.path.join(proj_cfg.METRICS_DIR, f"privacy_accounting_{self.client_id}.json")
            plot_path = os.path.join(proj_cfg.PLOTS_DIR, f"privacy_tradeoff_{self.client_id}.png")
            self.accountant.save_report(report_path)
            self.accountant.plot_tradeoff(plot_path)

            # FIX: Verify audit log integrity at end of training run
            print(f"[{self.client_id}] Verifying audit log integrity...")
            self.audit_log.verify_integrity()

        print(f"[{self.client_id}] Accuracy: {accuracy:.4f}  Loss: {loss:.4f}")
        return float(loss), self.num_samples, {"accuracy": float(accuracy)}


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, required=True)
    parser.add_argument("--use_dp", action="store_true")
    args = parser.parse_args()

    print(f"Starting client {args.client_id} (DP: {args.use_dp})...")
    fl.client.start_client(
        server_address=proj_cfg.SERVER_ADDRESS,
        client=FaceClient(args.client_id, args.use_dp).to_client(),
    )
