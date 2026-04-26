import os
import torch

EMBEDDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "local_embeddings")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def store_local_embedding(client_id: str, embedding: torch.Tensor):
    """Stores the embedding locally for Version 2 (Max Privacy)."""
    file_path = os.path.join(EMBEDDINGS_DIR, f"{client_id}.pt")
    torch.save(embedding, file_path)

def load_local_embedding(client_id: str):
    """Loads the local embedding for recognition."""
    file_path = os.path.join(EMBEDDINGS_DIR, f"{client_id}.pt")
    if os.path.exists(file_path):
        return torch.load(file_path)
    return None

def load_all_local_embeddings():
    """Loads all local embeddings to simulate device-level searching."""
    embeddings = {}
    if not os.path.exists(EMBEDDINGS_DIR):
        return embeddings
        
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".pt"):
            client_id = file.replace(".pt", "")
            embeddings[client_id] = torch.load(os.path.join(EMBEDDINGS_DIR, file))
    return embeddings
