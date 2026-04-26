import os
import shutil
import tempfile
import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch.nn.functional as F

from linking.client.client_service import process_and_register_locally, generate_embedding_only
from linking.client.local_storage import store_local_embedding, load_local_embedding
from linking.server.global_storage import store_global_embedding, load_all_global_embeddings
from linking.server.server_service import trigger_federated_learning
import config as proj_cfg

app = FastAPI(title="Privacy-Preserving Face Recognition FL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for Privacy Version
# 1 = Server stores embeddings
# 2 = Client stores embeddings locally
PRIVACY_VERSION = int(os.getenv("PRIVACY_VERSION", "1"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]

@app.post("/api/register")
async def register(name: str = Form(...), image: UploadFile = File(...)):
    if not image:
        raise HTTPException(status_code=400, detail="No image provided.")
        
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    try:
        # Save uploaded file temporarily
        file_path = os.path.join(temp_dir, image.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        image_paths.append(file_path)
            
        # 1. Preprocess and Generate Embedding
        # The user provided 'name', we treat it as client_id
        client_id = name.strip()
        embedding = process_and_register_locally(client_id, image_paths)
        
        # 2. Store Embedding based on Privacy Version
        if PRIVACY_VERSION == 1:
            # Version 1: Send to Server Storage
            store_global_embedding(client_id, embedding)
            storage_msg = "Embedding stored centrally on the server (Version 1)."
        elif PRIVACY_VERSION == 2:
            # Version 2: Store locally
            store_local_embedding(client_id, embedding)
            storage_msg = "Embedding stored locally on the client (Version 2)."
        else:
            raise ValueError(f"Unknown Privacy Version: {PRIVACY_VERSION}")
            
        return {
            "success": True, 
            "client_id": client_id, 
            "message": f"Face registered for '{client_id}' successfully! {storage_msg}"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Ensure we return a format the React app expects: {message: "..."}
        raise HTTPException(status_code=500, detail={"message": str(e)})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/api/recognize")
async def recognize(image: UploadFile = File(...), name: str = Form(None)):
    """
    name is required for Version 2 (local storage) to lookup the local file.
    For Version 1, it compares against all server embeddings.
    """
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, image.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        # 1. Generate embedding for the new face
        new_embedding = generate_embedding_only(file_path)
        
        # 2. Compare based on Privacy Version
        best_match = None
        best_similarity = -1.0
        
        if PRIVACY_VERSION == 1:
            # Version 1: Compare against global storage
            global_embeddings = load_all_global_embeddings()
            if not global_embeddings:
                return {"success": False, "message": "No registered users on the server."}
                
            for cid, emb in global_embeddings.items():
                sim = F.cosine_similarity(new_embedding, emb).item()
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = cid
                    
        elif PRIVACY_VERSION == 2:
            # Version 2: Compare against local storage
            from linking.client.local_storage import load_all_local_embeddings
            
            if name and name.strip():
                # Specific lookup
                client_id = name.strip()
                local_emb = load_local_embedding(client_id)
                if local_emb is None:
                    return {"success": False, "message": f"No local embedding found for {client_id}."}
                best_similarity = F.cosine_similarity(new_embedding, local_emb).item()
                best_match = client_id
            else:
                # Scan all local embeddings on device
                local_embeddings = load_all_local_embeddings()
                if not local_embeddings:
                    return {"success": False, "message": "No registered users stored locally on this device."}
                
                for cid, emb in local_embeddings.items():
                    sim = F.cosine_similarity(new_embedding, emb).item()
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = cid
            
        # 3. Decision
        if best_similarity >= proj_cfg.THRESHOLD:
            return {"success": True, "status": "recognized", "identity": best_match, "confidence": best_similarity}
        else:
            return {"success": True, "status": "unknown", "identity": None, "confidence": best_similarity}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"message": str(e)})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/api/federated-update")
async def federated_update(use_dp: bool = Form(False)):
    """Triggers the federated learning pipeline across all registered clients."""
    try:
        trigger_federated_learning(use_dp=use_dp)
        return {"success": True, "message": "Federated Learning round triggered in the background."}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": str(e)})

@app.get("/api/fl-results")
async def fl_results():
    """Returns FL results for the React dashboard."""
    metrics_dir = PROJECT_ROOT / "results" / "metrics"

    # Try DP file first, then no-DP
    for fname in ["fl_with_dp_results.json", "fl_no_dp_results.json"]:
        path = metrics_dir / fname
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            data["privacy_epsilon"] = 1.2 if data.get("dp") else "N/A"
            data["clients"]         = proj_cfg.MIN_CLIENTS
            data["model"]           = "InceptionResNetV1"
            return data

    # FL not run yet
    return {
        "rounds": [], "dp": False,
        "clients": 0,  "model": "Not trained yet",
        "privacy_epsilon": "N/A", "status": "pending"
    }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy", 
        "privacy_version": PRIVACY_VERSION,
        "privacy_mode": "Server Storage" if PRIVACY_VERSION == 1 else "Local Storage"
    }
