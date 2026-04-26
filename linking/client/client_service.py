import os
import torch
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader, Dataset
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocessing.detect import detect_face
from src.model.face_model import get_model, load_model
from src.federated.client import train_local
import config as proj_cfg

# Initialize MTCNN once globally for the API to reuse
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, keep_all=False, post_process=True)

class TempFaceDataset(Dataset):
    def __init__(self, tensor_files):
        self.files = tensor_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])

def get_latest_global_model(mode="train"):
    """Loads the latest federated global model if it exists, else returns the pretrained base."""
    # Check if a DP model exists, else NO_DP model, else fresh
    if os.path.exists(proj_cfg.MODEL_WITH_DP):
        return load_model(proj_cfg.MODEL_WITH_DP, mode=mode)
    elif os.path.exists(proj_cfg.MODEL_NO_DP):
        return load_model(proj_cfg.MODEL_NO_DP, mode=mode)
    else:
        return get_model(mode=mode)

def process_and_register_locally(client_id: str, image_paths: list) -> torch.Tensor:
    """
    1. Preprocesses images
    2. Saves them to the FL clients directory
    3. Generates and returns the embedding using the current global model
    """
    client_dir = os.path.join(proj_cfg.CLIENTS_DIR, f"client_{client_id}")
    os.makedirs(client_dir, exist_ok=True)
    
    saved_tensors = []
    
    # 1. Preprocess
    for idx, img_path in enumerate(image_paths):
        face_tensor = detect_face(img_path, mtcnn, check_blur=False)
        if face_tensor is not None:
            tensor_path = os.path.join(client_dir, f"face_{idx}.pt")
            torch.save(face_tensor, tensor_path)
            saved_tensors.append(tensor_path)
            
    if not saved_tensors:
        raise ValueError("No valid faces detected in the provided images.")
        
    # 2. Generate Embedding directly (no local epochs, FL handles training later)
    model = get_latest_global_model(mode="eval")
    with torch.no_grad():
        # Generate embedding using the first face as anchor (or average them)
        anchor_tensor = torch.load(saved_tensors[0]).unsqueeze(0)
        embedding = model(anchor_tensor)
        
    return embedding

def generate_embedding_only(image_path: str) -> torch.Tensor:
    """For recognition: just detect face and generate embedding using the global model."""
    face_tensor = detect_face(image_path, mtcnn, check_blur=False)
    if face_tensor is None:
        raise ValueError("No face detected for recognition.")
        
    model = get_latest_global_model(mode="eval")
    with torch.no_grad():
        embedding = model(face_tensor.unsqueeze(0))
    return embedding
