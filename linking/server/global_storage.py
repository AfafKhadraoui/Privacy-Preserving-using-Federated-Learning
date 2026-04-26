import os
import torch

SERVER_EMBEDDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "server_embeddings")

os.makedirs(SERVER_EMBEDDINGS_DIR, exist_ok=True)

def store_global_embedding(client_id: str, embedding: torch.Tensor):
    """Stores the embedding on the server for Version 1."""
    # The dictionary could be in memory, but we persist it for realistic behavior
    file_path = os.path.join(SERVER_EMBEDDINGS_DIR, f"{client_id}.pt")
    torch.save(embedding, file_path)

def load_all_global_embeddings():
    """Loads all embeddings for server-side recognition."""
    embeddings = {}
    if not os.path.exists(SERVER_EMBEDDINGS_DIR):
        return embeddings
        
    for file in os.listdir(SERVER_EMBEDDINGS_DIR):
        if file.endswith(".pt"):
            client_id = file.replace(".pt", "")
            embeddings[client_id] = torch.load(os.path.join(SERVER_EMBEDDINGS_DIR, file))
    return embeddings
