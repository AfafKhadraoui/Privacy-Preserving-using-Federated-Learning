
import os
import argparse
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import flwr as fl

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as proj_cfg
from src.model.face_model import get_model, get_parameters, set_parameters



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
        # Walk all subfolders (person name folders) and collect .pt files
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
        return tensor                           # no label needed — see loss below


# =============================================================================
# Training with embedding-based loss
# =============================================================================

def train_local(model, loader, optimizer, epochs):
    """
    Train the model locally using embedding consistency loss.

    Since Kosai's model outputs 512-dim embeddings (not class scores),
    we use a self-supervised approach:
        1. Take each face image, get its embedding
        2. Apply a small random augmentation, get another embedding
        3. Loss = MSE between the two embeddings (they should be similar)
    This teaches the model that the same face should always produce
    a similar embedding regardless of small variations.

    We do NOT use CrossEntropyLoss here because the model has no 
    classification head — it outputs embeddings directly.
    """
    from src.augmentation.augment import FaceAugmentation

    # Use the team's dedicated Augmentation module!
    augment = FaceAugmentation(use_affine=True, use_blur=False)

    model.train()
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            # batch is a list of tensors, shape [B, 3, 160, 160]
            images = batch
            
            # CRITICAL FIX: PyTorch BatchNorm crashes if it receives a batch of exactly 1 image.
            # If this client only has 1 photo, we silently duplicate it in the batch.
            if len(images) == 1:
                images = torch.cat([images, images], dim=0)

            optimizer.zero_grad()

            # View 1: original embedding
            emb1 = model(images)

            # View 2: augmented version of the same images
            images_aug = torch.stack([augment(img) for img in images])
            emb2 = model(images_aug)

            # Loss: embeddings of the same face should be close
            loss = criterion(emb1, emb2)
            loss.backward()
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

    # Use the team's dedicated Augmentation module!
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
    # Convert loss to pseudo-accuracy (loss near 0 = accuracy near 1)
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
    """

    def __init__(self, client_id, use_dp):
        self.client_id = f"client_{client_id:02d}"
        self.client_dir = os.path.join(proj_cfg.CLIENTS_DIR, self.client_id)
        self.use_dp = use_dp

        # Load Kosai's model — embedding mode (no classification head)
        self.model = get_model(mode="train")

        # Build dataset
        dataset = FaceDataset(self.client_dir)

        if len(dataset) == 0:
            print(f"[{self.client_id}] ERROR: No photos found in {self.client_dir}. Terminating this simulated client.")
            import sys
            sys.exit(0)
        else:
            self.num_samples = len(dataset)

        # collate_fn=lambda x: torch.stack(x) because dataset returns tensors not tuples
        self.train_loader = DataLoader(
            dataset, batch_size=8, shuffle=True,
            collate_fn=lambda x: torch.stack(x)
        )
        self.eval_loader = DataLoader(
            dataset, batch_size=8, shuffle=False,
            collate_fn=lambda x: torch.stack(x)
        )

        print(f"\n[{self.client_id}] Initialization successful! Ready to connect to Server.")
        print(f"  Photos: {self.num_samples}")
        print(f"  DP:     {'ON  → Version B' if use_dp else 'OFF → Version A'}")

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
            4. Return updated weights + num_samples to server
        """

        # Step 1 — load global weights
        set_parameters(self.model, parameters)

        # Step 2 — create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=proj_cfg.LEARNING_RATE)

        # Step 3 — train (this is where A and B split)
        if self.use_dp:
            # VERSION B: Uses Differential Privacy Module
            from src.privacy.dp_training import make_private, get_epsilon
            
            privacy_engine, model_dp, optimizer_dp, loader_dp = make_private(
                model=self.model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                noise_multiplier=proj_cfg.NOISE_MULTIPLIER,
                max_grad_norm=proj_cfg.MAX_GRAD_NORM,
            )
            print(f"[{self.client_id}] Training WITH DP (Safe Mode)...")
            train_local(model_dp, loader_dp, optimizer_dp, proj_cfg.LOCAL_EPOCHS)
            epsilon = get_epsilon(privacy_engine, delta=proj_cfg.DELTA)
            print(f"[{self.client_id}] Privacy budget spent: ε = {epsilon:.2f}")
        else:
            # VERSION A: plain training, no protection
            print(f"[{self.client_id}] Training WITHOUT DP...")
            train_local(self.model, self.train_loader, optimizer, proj_cfg.LOCAL_EPOCHS)

        # Step 4 — return updated weights
        return get_parameters(self.model), self.num_samples, {}

    def evaluate(self, parameters, config):
        """
        Measure how good the current global model is on local data.
        Returns real accuracy (not hardcoded 1.0).
        """
        set_parameters(self.model, parameters)
        loss, accuracy = evaluate_local(self.model, self.eval_loader)
        print(f"[{self.client_id}] Accuracy: {accuracy:.4f}  Loss: {loss:.4f}")
        return float(loss), self.num_samples, {"accuracy": float(accuracy)}


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--use_dp", action="store_true")
    args = parser.parse_args()

    print(f"Starting client {args.client_id} (DP: {args.use_dp})...")
    fl.client.start_client(
        server_address=proj_cfg.SERVER_ADDRESS,
        client=FaceClient(args.client_id, args.use_dp).to_client(),
    )