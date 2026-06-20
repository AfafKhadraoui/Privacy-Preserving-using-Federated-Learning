"""
GLOBAL FACE MODEL MODULE (Federated Learning Core)

This file defines the main face recognition model used across the entire
federated learning system.

MODEL ARCHITECTURE:
- Uses InceptionResnetV1 (FaceNet-style backbone)
- Pretrained on VGGFace2 for robust face embeddings
- Produces fixed-length feature vectors (embeddings) instead of class labels

ROLE IN THE SYSTEM:
This model acts as the GLOBAL MODEL in the federated learning pipeline:
- Initialized on the server
- Sent to all clients for local training
- Updated locally on client devices using private datasets
- Aggregated on the server using Federated Averaging (FedAvg)

FEDERATED LEARNING UTILITIES INCLUDED:
- get_model(): initializes pretrained model for training or evaluation
- get_parameters(): extracts model weights as numpy arrays (Flower-compatible)
- set_parameters(): loads aggregated weights back into the model
- save_model(): saves global model weights to disk for persistence across rounds
- load_model(): loads previously saved model from disk or initializes a new one if not found
- federated_averaging(): aggregates multiple client model updates using (weighted) averaging to produce the new global model

OUTPUT:
- Produces face embeddings (not classifications)
- These embeddings are used for similarity-based face recognition
  (e.g., cosine similarity or Euclidean distance)

USAGE FLOW:
Server:
    initialize or load global model → distribute to clients → aggregate updates using FedAvg

Client:
    receive model → train locally → send updated parameters

Server:
    aggregate updates → update global model → save model → redistribute
"""
import os
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1


# -----------------------------
# Global Model Definition
# -----------------------------
class GlobalFaceModel(torch.nn.Module):
    """
    FaceNet-style global model using InceptionResnetV1.
    Designed for Federated Learning (Flower-compatible).
    """

    def __init__(self, pretrained="vggface2"):
        super(GlobalFaceModel, self).__init__()

        self.model = InceptionResnetV1(
            pretrained=pretrained,
            classify=False
        )
 
    def forward(self, x):
        return self.model(x)


# -----------------------------
# Model Loader
# -----------------------------
def get_model(mode="train"):
    """
    Load pretrained InceptionResnetV1 in embedding mode.
    """
    model = GlobalFaceModel(pretrained="vggface2")

    if mode == "eval":
        model.eval()

    return model


# -----------------------------
# FL Utility: Model → Parameters
# -----------------------------
def get_parameters(model):
    """
    Convert model weights to list of numpy arrays (Flower format).
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


# -----------------------------
# FL Utility: Parameters → Model
# -----------------------------
def set_parameters(model, parameters):
    """
    Load numpy arrays back into the model state_dict.
    Zips incoming arrays with model keys.
    """
    state_dict = model.state_dict()
    new_state_dict = {}

    for (key, _), param in zip(state_dict.items(), parameters):
        new_state_dict[key] = torch.tensor(param)

    model.load_state_dict(new_state_dict, strict=True)
    return model

# -----------------------------
# FL Utility: Federated Averaging (Client Updates → Global Model)
# -----------------------------
def federated_averaging(client_parameters, client_sizes=None):
    """
    Perform Federated Averaging (FedAvg) on client model parameters.

    Args:
        client_parameters (list):
            List of model parameter lists (each client update).
            Example: [client1_params, client2_params, ...]

        client_sizes (list, optional):
            Number of samples per client. Used for weighted averaging.
            If None → simple average is used.

    Returns:
        List of aggregated parameters (global model weights).
    """

    # Number of clients
    num_clients = len(client_parameters)

    # Force float dtype to avoid int/float conflicts
    aggregated = [
        np.zeros_like(param, dtype=np.float64)
        for param in client_parameters[0]
    ]

    # -----------------------------
    # Simple averaging (unweighted)
    # -----------------------------
    if client_sizes is None:
        for params in client_parameters:
            for i, param in enumerate(params):
                aggregated[i] += param.astype(np.float64)

        aggregated = [p / num_clients for p in aggregated]

    # -----------------------------
    # Weighted averaging
    # -----------------------------
    else:
        total_samples = float(sum(client_sizes))

        for params, size in zip(client_parameters, client_sizes):
            weight = size / total_samples

            for i, param in enumerate(params):
                aggregated[i] += param.astype(np.float64) * weight

    return aggregated


# -----------------------------
# Model Persistence (Save / Load)
# -----------------------------
def save_model(model, path):
    """
    Save model weights to a file.

    Used on server to persist global model between FL rounds.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path, mode="train"):
    """
    Load model weights from file.

    If file does not exist, returns a new initialized model.
    """
    model = GlobalFaceModel(pretrained="vggface2")

    if os.path.exists(path):
        state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        model.load_state_dict(state_dict)
    else:
        # No saved model → initialize fresh
        print(f"[INFO] No saved model found at {path}. Initializing new model.")

    if mode == "eval":
        model.eval()

    return model