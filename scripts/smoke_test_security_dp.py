"""
Lightweight smoke tests for privacy/security modules without full FL training.

What this script checks:
1) X25519 key exchange + HKDF key derivation
2) AES-256-GCM payload encryption/decryption with AAD
3) Ed25519 signing + verification (valid and tampered)
4) DP wrapper behavior on synthetic data (epsilon tracking + budget exception)
5) Existing project FL readiness precheck (folders/dependencies/API capability)

Run:
    python scripts/smoke_test_security_dp.py
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import os
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import flwr as fl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.privacy.payload_encryption import (
    PayloadDecryptor,
    PayloadEncryptor,
    compute_shared_secret,
    derive_symmetric_key,
    generate_ephemeral_keypair,
)
from src.privacy.signing import (
    SignatureValidator,
    generate_client_keypair,
    sign_update,
    verify_update,
)
from src.privacy.dp_training import (
    PrivacyBudgetExceeded,
    make_private_with_dp,
)


def hash_parameters(ndarrays) -> str:
    """Create a stable SHA-256 digest for a parameter payload."""
    hasher = hashlib.sha256()
    for arr in ndarrays:
        hasher.update(arr.tobytes())
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
    return hasher.hexdigest()


class TinyNet(nn.Module):
    """Tiny model used only for the federated smoke pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


def tiny_get_parameters(model: nn.Module):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def tiny_set_parameters(model: nn.Module, parameters):
    state_dict = model.state_dict()
    new_state_dict = {}
    for (key, _), param in zip(state_dict.items(), parameters):
        new_state_dict[key] = torch.tensor(param)
    model.load_state_dict(new_state_dict, strict=True)
    return model


def build_tiny_dataset(seed: int, num_samples: int = 32):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_samples, 4, generator=generator)
    labels = ((features.sum(dim=1) > 0).long() + (seed % 2)) % 2
    return DataLoader(TensorDataset(features, labels), batch_size=8, shuffle=True)


class TinySmokeClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, train_loader: DataLoader, eval_loader: DataLoader, key_dir: str):
        self.client_id = client_id
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = TinyNet()
        self.private_key, self.public_key = generate_client_keypair(client_id, output_dir=key_dir)

    def get_parameters(self, config):
        return tiny_get_parameters(self.model)

    def fit(self, parameters, config):
        tiny_set_parameters(self.model, parameters)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        private_model, private_optimizer, private_loader, engine, monitor = make_private_with_dp(
            model=self.model,
            optimizer=optimizer,
            train_loader=self.train_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            delta=1e-5,
            epsilon_max=10.0,
            random_seed=123,
            client_id=self.client_id,
        )

        for xb, yb in private_loader:
            private_optimizer.zero_grad()
            logits = private_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            private_optimizer.step()

        epsilon = monitor.check_and_log(engine)
        updated_parameters = tiny_get_parameters(private_model)
        model_hash = hash_parameters(updated_parameters)

        update_dict = {
            "round_id": int(config.get("server_round", 1)),
            "client_id": self.client_id,
            "num_samples": len(self.train_loader.dataset),
            "model_hash": model_hash,
            "protocol_version": "v1",
        }
        signature = sign_update(self.private_key, update_dict)
        public_key_raw = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        metrics = {
            "client_id": self.client_id,
            "round_id": int(config.get("server_round", 1)),
            "epsilon": float(epsilon),
            "model_hash": model_hash,
            "num_samples": len(self.train_loader.dataset),
            "protocol_version": "v1",
            "signature_hex": signature.hex(),
            "public_key_raw_b64": base64.b64encode(public_key_raw).decode("ascii"),
        }
        return updated_parameters, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        tiny_set_parameters(self.model, parameters)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        with torch.no_grad():
            for xb, yb in self.eval_loader:
                logits = self.model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * len(yb)
                predictions = logits.argmax(dim=1)
                total_correct += (predictions == yb).sum().item()
                total_examples += len(yb)

        accuracy = total_correct / total_examples if total_examples else 0.0
        return total_loss / total_examples if total_examples else 0.0, total_examples, {"accuracy": accuracy}


class ValidatingFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            client_id = metrics.get("client_id")
            signature_hex = metrics.get("signature_hex")
            public_key_raw_b64 = metrics.get("public_key_raw_b64")
            model_hash = metrics.get("model_hash")

            if not client_id or not signature_hex or not public_key_raw_b64 or not model_hash:
                raise RuntimeError("Missing signature metadata from client update")

            public_key = ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_raw_b64))
            update_dict = {
                "round_id": int(metrics.get("round_id", server_round)),
                "client_id": client_id,
                "num_samples": int(metrics.get("num_samples", 0)),
                "model_hash": model_hash,
                "protocol_version": metrics.get("protocol_version", "v1"),
            }
            if not verify_update(public_key, bytes.fromhex(signature_hex), update_dict):
                raise RuntimeError(f"Signature verification failed for {client_id}")

            print(f"[FL Smoke] Verified signed update from {client_id}")

        return super().aggregate_fit(server_round, results, failures)


def run_fl_server(address: str, num_rounds: int = 1):
    strategy = ValidatingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


def run_fl_client(client_index: int, address: str, key_dir: str):
    client_id = f"smoke_client_{client_index:02d}"
    train_loader = build_tiny_dataset(seed=100 + client_index, num_samples=32)
    eval_loader = build_tiny_dataset(seed=200 + client_index, num_samples=16)
    client = TinySmokeClient(client_id, train_loader, eval_loader, key_dir)
    fl.client.start_numpy_client(server_address=address, client=client)


def test_actual_fl_pipeline() -> None:
    print("\n=== Test 4: Real Flower FL Pipeline on Synthetic Data ===")

    address = "127.0.0.1:8091"
    num_clients = 2

    with tempfile.TemporaryDirectory() as tmp:
        key_dir = os.path.join(tmp, "keys")
        os.makedirs(key_dir, exist_ok=True)

        server_process = mp.Process(target=run_fl_server, args=(address, 1), daemon=True)
        server_process.start()

        time.sleep(1.5)

        client_processes = []
        for client_index in range(num_clients):
            process = mp.Process(target=run_fl_client, args=(client_index, address, key_dir), daemon=True)
            process.start()
            client_processes.append(process)

        for process in client_processes:
            process.join(timeout=60)
            if process.exitcode not in (0, None):
                raise RuntimeError(f"FL smoke client exited with code {process.exitcode}")

        server_process.join(timeout=60)
        if server_process.exitcode not in (0, None):
            raise RuntimeError(f"FL smoke server exited with code {server_process.exitcode}")

    _ok("Flower server/client exchange completed on synthetic data")


def _count_pt_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.rglob("*.pt"))


def test_existing_project_fl_precheck() -> None:
    """Validate whether the existing project FL pipeline can run with current folders/deps."""
    print("\n=== Test 4: Existing Project FL Precheck ===")

    import config

    clients_dir = Path(config.CLIENTS_DIR)
    cropped_dir = Path(config.CROPPED_DIR)
    raw_dir = Path(config.DATA_DIR)

    issues = []

    if not raw_dir.exists():
        issues.append(f"Missing raw data folder: {raw_dir}")
    if not cropped_dir.exists():
        issues.append(f"Missing cropped data folder: {cropped_dir}")
    if not clients_dir.exists():
        issues.append(f"Missing clients data folder: {clients_dir}")

    cropped_pt_count = _count_pt_files(cropped_dir)
    client_folders = sorted([p for p in clients_dir.glob("client_*") if p.is_dir()]) if clients_dir.exists() else []
    client_pt_count = _count_pt_files(clients_dir)

    print(f"[Precheck] raw_dir exists: {raw_dir.exists()}")
    print(f"[Precheck] cropped_dir exists: {cropped_dir.exists()} | .pt files: {cropped_pt_count}")
    print(f"[Precheck] clients_dir exists: {clients_dir.exists()} | client folders: {len(client_folders)} | .pt files: {client_pt_count}")

    min_clients = int(getattr(config, "MIN_CLIENTS", 1))
    if len(client_folders) < min_clients:
        issues.append(f"Not enough client folders for config.MIN_CLIENTS={min_clients}")

    if importlib.util.find_spec("facenet_pytorch") is not None:
        print("[Precheck] facenet_pytorch import: OK")
    else:
        issues.append("facenet_pytorch is not installed in current environment")

    from src.privacy.secure_agg import has_legacy_secagg_strategy, has_workflow_secagg

    legacy_secagg_available = has_legacy_secagg_strategy()
    workflow_secagg_available = has_workflow_secagg()
    print(f"[Precheck] Legacy SecAggPlusStrategy available: {legacy_secagg_available}")
    print(f"[Precheck] Workflow+Mod SecAgg API available: {workflow_secagg_available}")

    if not legacy_secagg_available and workflow_secagg_available:
        print("[Precheck] INFO: Current Flower uses workflow/mod API for secure aggregation")
        print("[Precheck] INFO: Existing src/federated launcher can train, but true SecAgg requires ServerApp/ClientApp migration")
    elif not workflow_secagg_available:
        print("[Precheck] WARNING: No supported Flower secure aggregation API detected")

    if issues:
        print("[Precheck] Existing FL pipeline is NOT ready yet:")
        for issue in issues:
            print(f"  - {issue}")
        return

    _ok("Existing project FL folders/dependencies look ready")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def test_crypto_handshake_and_payload() -> None:
    print("\n=== Test 1: Key Exchange + Payload Encryption ===")

    # Server long-term X25519 key pair.
    from cryptography.hazmat.primitives.asymmetric import x25519

    server_private = x25519.X25519PrivateKey.generate()
    server_public = server_private.public_key()
    server_private_bytes = server_private.private_bytes_raw()
    server_public_bytes = server_public.public_bytes_raw()

    print("Server X25519 public key:", server_public_bytes.hex())
    print("Server X25519 private key:", server_private_bytes.hex())

    # Client ephemeral key pair and shared secret.
    client_private_bytes, client_public_bytes = generate_ephemeral_keypair()
    client_shared = compute_shared_secret(client_private_bytes, server_public_bytes)
    server_shared = compute_shared_secret(server_private_bytes, client_public_bytes)

    if client_shared != server_shared:
        _fail("X25519 shared secrets do not match")
    _ok("X25519 shared secrets match")

    client_aes_key = derive_symmetric_key(client_shared, info="model_encryption")
    server_aes_key = derive_symmetric_key(server_shared, info="model_encryption")

    if client_aes_key != server_aes_key:
        _fail("HKDF-derived AES keys do not match")
    _ok("HKDF-derived AES keys match")

    print("Derived AES-256 key:", client_aes_key.hex())

    payload = b"hello from client payload"
    round_id = 1
    client_id = "client_00"
    protocol_version = "v1"

    encryptor = PayloadEncryptor(server_public_bytes)
    decryptor = PayloadDecryptor(server_private_bytes)

    encrypted = encryptor.encrypt(
        plaintext=payload,
        round_id=round_id,
        client_id=client_id,
        protocol_version=protocol_version,
    )

    print("Ephemeral public key:", encrypted["ephemeral_public_key"])
    print("Encrypted payload (hex):", encrypted["encrypted_payload"])
    print("Nonce (first 12 bytes):", encrypted["encrypted_payload"][:24])

    recovered = decryptor.decrypt(encrypted)
    if recovered != payload:
        _fail("Decrypt output does not match original payload")
    _ok("AES-256-GCM encrypt/decrypt with AAD succeeded")

    # Tamper AAD metadata to prove integrity check works.
    tampered = dict(encrypted)
    tampered["client_id"] = "client_01"

    tamper_failed = False
    try:
        decryptor.decrypt(tampered)
    except Exception:
        tamper_failed = True

    if not tamper_failed:
        _fail("Tampered AAD unexpectedly decrypted")
    _ok("AAD tampering correctly rejected")


def test_signing() -> None:
    print("\n=== Test 2: Signing + Verification ===")

    with tempfile.TemporaryDirectory() as tmp:
        private_key, public_key = generate_client_keypair("client_00", output_dir=tmp)

        update = {
            "round_id": 1,
            "client_id": "client_00",
            "num_samples": 8,
            "model_hash": "abc123",
            "protocol_version": "v1",
        }

        signature = sign_update(private_key, update)
        print("Signature (hex):", signature.hex())

        if not verify_update(public_key, signature, update):
            _fail("Valid signature rejected")
        _ok("Direct verify_update accepted valid signature")

        tampered = dict(update)
        tampered["model_hash"] = "evil"
        if verify_update(public_key, signature, tampered):
            _fail("Tampered update passed signature verification")
        _ok("Tampered update rejected")

        # Server-side validator path.
        validator = SignatureValidator()
        validator.register_client_key("client_00", os.path.join(tmp, "client_00_public.pem"))
        if not validator.validate_update("client_00", signature, update):
            _fail("SignatureValidator rejected valid signature")
        _ok("SignatureValidator accepted valid signature")


def test_dp_smoke() -> None:
    print("\n=== Test 3: DP Smoke Test (Synthetic Data) ===")

    torch.manual_seed(42)

    X = torch.randn(64, 10)
    y = torch.randint(0, 2, (64,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)

    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    model_dp, opt_dp, loader_dp, engine, monitor = make_private_with_dp(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        delta=1e-5,
        epsilon_max=10.0,
        random_seed=42,
        client_id="smoke_client",
    )

    for _ in range(2):
        for xb, yb in loader_dp:
            opt_dp.zero_grad()
            logits = model_dp(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt_dp.step()

    eps = monitor.check_and_log(engine)
    print(f"DP epsilon after tiny synthetic run: {eps:.4f}")
    _ok("DP wrapper + epsilon monitoring works")

    # Confirm budget enforcement exception path.
    strict_model = nn.Sequential(nn.Linear(10, 2))
    strict_opt = torch.optim.SGD(strict_model.parameters(), lr=0.05)

    strict_model, strict_opt, strict_loader, strict_engine, strict_monitor = make_private_with_dp(
        model=strict_model,
        optimizer=strict_opt,
        train_loader=loader,
        noise_multiplier=0.1,
        max_grad_norm=1.0,
        delta=1e-5,
        epsilon_max=0.001,
        random_seed=42,
        client_id="strict_budget_client",
    )

    budget_triggered = False
    try:
        for xb, yb in strict_loader:
            strict_opt.zero_grad()
            logits = strict_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            strict_opt.step()
        strict_monitor.check_and_log(strict_engine)
    except PrivacyBudgetExceeded:
        budget_triggered = True

    if not budget_triggered:
        _fail("Expected PrivacyBudgetExceeded was not raised")
    _ok("Privacy budget enforcement path works")


def main() -> None:
    parser = argparse.ArgumentParser(description="Privacy/security smoke tests")
    parser.add_argument(
        "--run-synthetic-fl",
        action="store_true",
        help="Run the synthetic Flower server/client exchange test (can take longer)",
    )
    args = parser.parse_args()

    print("Running lightweight privacy/security smoke tests...")
    test_crypto_handshake_and_payload()
    test_signing()
    test_dp_smoke()
    test_existing_project_fl_precheck()
    if args.run_synthetic_fl:
        test_actual_fl_pipeline()
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
