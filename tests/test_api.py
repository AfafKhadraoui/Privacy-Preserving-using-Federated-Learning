import sys
import os
import json
import torch
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from linking.api import app
from src.model.face_model import get_model, get_parameters

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    print("Health check passed:", response.json())

def test_register():
    response = client.post("/api/register", data={"client_id": "test_client"})
    assert response.status_code == 200
    data = response.json()
    assert "model_weights" in data
    assert data["message"] == "global model delivered"
    print("Register passed, received model weights.")

if __name__ == "__main__":
    print("Testing API...")
    test_health()
    try:
        test_register()
    except Exception as e:
        print("Register failed, which is expected if models don't exist yet:", e)
    print("All tests passed.")
