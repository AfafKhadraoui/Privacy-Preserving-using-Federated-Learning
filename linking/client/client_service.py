import os
import torch
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader, Dataset
import sys
import requests
import io
import base64
import numpy as np
import json

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocessing.detect import detect_face
from src.model.face_model import get_model, set_parameters
from linking.client.local_storage import store_local_embedding, load_all_local_embeddings
import torch.nn.functional as F
import config as proj_cfg

# Initialize MTCNN once globally for the client to reuse
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, keep_all=False, post_process=True)

API_BASE_URL = "http://127.0.0.1:5000"

class TempFaceDataset(Dataset):
    def __init__(self, tensor_files):
        self.files = tensor_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])

def fetch_global_model(client_id: str, mode="eval"):
    """
    Client calls /api/register to receive the global model weights.
    Loads them into the PyTorch model and returns it.
    """
    response = requests.post(f"{API_BASE_URL}/api/register", data={"client_id": client_id})
    if response.status_code == 200:
        data = response.json()
        b64_weights = data["model_weights"]
        buffer = io.BytesIO(base64.b64decode(b64_weights))
        npz = np.load(buffer)
        parameters = [npz[key] for key in npz.files]
        
        model = get_model(mode=mode)
        set_parameters(model, parameters)
        return model
    else:
        raise Exception(f"Failed to fetch global model from API: {response.text}")

def process_and_register_locally(client_id: str, image_paths: list) -> torch.Tensor:
    """
    1. Preprocesses images and saves them locally for FL training.
    2. Calls /api/register to get the global model.
    3. Generates embeddings locally using the global model and averages them.
    4. Calls /api/register_update to send the embedding and trigger FL.
    """
    client_id = client_id.strip()
    if not image_paths:
        raise ValueError("At least one registration image is required.")

    client_dir = os.path.join(proj_cfg.CLIENTS_DIR, f"client_{client_id}")
    os.makedirs(client_dir, exist_ok=True)
    
    saved_tensors = []
    face_tensors = []
    
    # 1. Preprocess
    for img_path in image_paths:
        face_tensor = detect_face(img_path, mtcnn, check_blur=False)
        if face_tensor is not None:
            face_tensors.append(face_tensor)
            
    if len(face_tensors) != len(image_paths):
        raise ValueError(
            f"Detected valid faces in {len(face_tensors)} of {len(image_paths)} registration images."
        )

    for file_name in os.listdir(client_dir):
        if file_name.startswith("face_") and file_name.endswith(".pt"):
            os.remove(os.path.join(client_dir, file_name))

    for idx, face_tensor in enumerate(face_tensors):
        tensor_path = os.path.join(client_dir, f"face_{idx}.pt")
        torch.save(face_tensor, tensor_path)
        saved_tensors.append(tensor_path)
        
    # 2. Fetch global model via API instead of reading from disk
    model = fetch_global_model(client_id, mode="eval")
    
    # 3. Generate embeddings locally and aggregate them for registration.
    with torch.no_grad():
        face_batch = torch.stack(face_tensors)
        embeddings = model(face_batch)
        embedding = embeddings.mean(dim=0)

    # Store locally for V2
    store_local_embedding(client_id, embedding)

    # Get privacy version
    res_health = requests.get(f"{API_BASE_URL}/api/health")
    priv_ver = res_health.json().get("privacy_version", 1) if res_health.status_code == 200 else 1

    # 4. Send update to API (triggering FL, sending embedding only if V1)
    payload = {
        "client_id": client_id,
        "num_samples": len(saved_tensors)
    }
    if priv_ver == 1:
        payload["embedding"] = embedding.tolist()

    res = requests.post(f"{API_BASE_URL}/api/register_update", data={"payload": json.dumps(payload)})
    if res.status_code != 200:
        raise Exception(f"Failed to register update with API: {res.text}")
        
    return embedding

def generate_embedding_only(image_path: str) -> torch.Tensor:
    """
    For recognition: detects face and generates embedding locally
    using the global model fetched from the API.
    """
    face_tensor = detect_face(image_path, mtcnn, check_blur=False)
    if face_tensor is None:
        raise ValueError("No face detected for recognition.")
        
    # Fetch model via API
    model = fetch_global_model("recognize_client", mode="eval")
    
    with torch.no_grad():
        embedding = model(face_tensor.unsqueeze(0))
        
    return embedding

def recognize_face(image_path: str):
    """
    End-to-end recognition test:
    1. Extract embedding locally.
    2. Check privacy version.
    3. If V1, send embedding to /api/recognize.
    4. If V2, perform cosine similarity locally against stored embeddings.
    """
    embedding = generate_embedding_only(image_path)
    
    res_health = requests.get(f"{API_BASE_URL}/api/health")
    priv_ver = res_health.json().get("privacy_version", 1) if res_health.status_code == 200 else 1

    if priv_ver == 1:
        payload = {
            "embedding": embedding.squeeze().tolist()
        }
        response = requests.post(f"{API_BASE_URL}/api/recognize", data={"payload": json.dumps(payload)})
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API recognition failed: {response.text}")
    else:
        # Local Recognition for V2
        local_embeddings = load_all_local_embeddings()
        if not local_embeddings:
            return {
                "status": "unknown",
                "identity": "unknown",
                "confidence": 0.0
            }

        best_match = "unknown"
        best_sim = -1.0
        
        for cid, emb in local_embeddings.items():
            emb = emb.view(-1)
            client_emb_flat = embedding.view(-1)
            
            sim = F.cosine_similarity(client_emb_flat, emb, dim=0).item()
            if sim > best_sim:
                best_sim = sim
                best_match = cid

        if best_sim >= proj_cfg.THRESHOLD:
            return {
                "status": "recognized",
                "identity": best_match,
                "confidence": float(best_sim)
            }
        else:
            return {
                "status": "unknown",
                "identity": "unknown",
                "confidence": float(best_sim)
            }
