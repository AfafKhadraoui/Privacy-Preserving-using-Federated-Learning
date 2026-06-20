import io
import os
import json
import base64
import torch
import numpy as np
import torch.nn.functional as F

from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config as proj_cfg
from src.model.face_model import load_model, get_parameters
from linking.server.global_storage import load_all_global_embeddings, store_global_embedding
from linking.server.server_service import trigger_federated_learning

app = FastAPI(title="FL Face System (Model-Centric API)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PRIVACY_VERSION = int(os.getenv("PRIVACY_VERSION", "1"))

def get_global_model():
    """
    Loads the correct global model (with DP or no DP based on config),
    extracts the weights as numpy arrays, and serializes them as base64.
    """
    model_path = proj_cfg.MODEL_WITH_DP if proj_cfg.USE_DP else proj_cfg.MODEL_NO_DP
    model = load_model(model_path)
    parameters = get_parameters(model)
    
    # Serialize parameters (list of numpy arrays) using np.savez_compressed
    buffer = io.BytesIO()
    np.savez_compressed(buffer, *parameters)
    b64_weights = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return {
        "weights": b64_weights,
        "version": proj_cfg.PROTOCOL_VERSION
    }


@app.post("/api/register")
async def register(client_id: str = Form(...)):
    """
    FLOW:
    1. Client calls /api/register
    2. Server responds with global model parameters and version
    3. Client receives model and trains locally (no training in API)
    """
    try:
        global_model_data = get_global_model()

        return {
            "model_weights": global_model_data["weights"],
            "model_version": global_model_data["version"],
            "message": "global model delivered"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/register_update")
async def register_update(payload: str = Form(...)):
    """
    FLOW:
    Client sends updated model weights, optional embedding, etc.
    Server validates, forwards to FL pipeline, and stores embedding if V1.
    """
    try:
        data = json.loads(payload)
        client_id = data["client_id"]
        
        # Store embedding if PRIVACY_VERSION == 1
        embedding_list = data.get("embedding")
        if PRIVACY_VERSION == 1 and embedding_list is not None:
            embedding_tensor = torch.tensor(embedding_list, dtype=torch.float32)
            store_global_embedding(client_id, embedding_tensor)

        # Trigger FL round on the server side
        trigger_federated_learning(use_dp=proj_cfg.USE_DP)

        return {
            "status": "update_received",
            "fl_status": "queued"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recognize")
async def recognize(payload: str = Form(...)):
    """
    FLOW:
    1. Client sends embedding
    2. Server compares embeddings using cosine similarity
    3. Returns identity result
    """
    try:
        data = json.loads(payload)
        client_emb_list = data.get("embedding")
        
        if not client_emb_list:
            raise HTTPException(status_code=400, detail="Missing embedding in payload")
            
        client_emb = torch.tensor(client_emb_list, dtype=torch.float32)

        if PRIVACY_VERSION == 1:
            server_embeddings = load_all_global_embeddings()
            if not server_embeddings:
                return {
                    "status": "unknown",
                    "identity": "unknown",
                    "confidence": 0.0
                }

            best_match = "unknown"
            best_sim = -1.0
            
            for cid, emb in server_embeddings.items():
                emb = emb.view(-1)
                client_emb_flat = client_emb.view(-1)
                
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
        else:
            # For V2, the server doesn't store embeddings
            return {
                "status": "error",
                "message": "Server recognition disabled in V2. Local recognition required."
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/federated-update")
async def federated_update(use_dp: bool = False):
    """
    ONLY responsibility: start FL round on server
    """
    trigger_federated_learning(use_dp=use_dp)

    return {
        "status": "FL_started",
        "dp": use_dp
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "architecture": "model-centric-fl-api",
        "privacy_version": PRIVACY_VERSION
    }

@app.get("/api/fl-results")
async def get_fl_results():
    """
    Returns mock FL results for the React frontend FLResults component.
    """
    import os
    import json
    
    rounds = []
    status = "pending"
    metrics_file = os.path.join(proj_cfg.METRICS_DIR, "fl_with_dp_results.json" if proj_cfg.USE_DP else "fl_no_dp_results.json")
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
                if data.get("success"):
                    status = "completed"
                    # Generate some dummy data since the actual results file doesn't have round-by-round stats yet
                    for i in range(1, proj_cfg.NUM_ROUNDS + 1):
                        rounds.append({
                            "round": i,
                            "accuracy": 0.5 + (0.4 * (i / proj_cfg.NUM_ROUNDS)),
                            "loss": 2.0 - (1.5 * (i / proj_cfg.NUM_ROUNDS))
                        })
        except:
            pass

    return {
        "status": status,
        "rounds": rounds,
        "clients": proj_cfg.MIN_CLIENTS,
        "num_rounds": proj_cfg.NUM_ROUNDS,
        "privacy_epsilon": proj_cfg.EPSILON_MAX,
        "dp": proj_cfg.USE_DP
    }

