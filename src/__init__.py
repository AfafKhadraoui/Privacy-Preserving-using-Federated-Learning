from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import json
from pathlib import Path

# ── Project root setup ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config as proj_cfg
from src.preprocessing.detect import detect_face
from facenet_pytorch import MTCNN
import torch

app = Flask(__name__)
CORS(app)  # allow React on :3000 to call Flask on :5000

# Init MTCNN once at startup (expensive — don't move inside route)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40,
              keep_all=False, post_process=True, device=device)


@app.route('/api/register', methods=['POST'])
def register():
    name  = request.form.get('name', '').strip()
    file  = request.files.get('image')

    if not name or not file:
        return jsonify({"success": False, "message": "Name and image are required"}), 400

    # ── Step 1: Save raw image to data/raw/<name>/ ─────────────────────────────
    raw_dir = PROJECT_ROOT / "data" / "raw" / name
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Count existing photos to auto-number them (photo_1.jpg, photo_2.jpg ...)
    existing = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png"))
    ext      = Path(file.filename).suffix or ".jpg"
    raw_path = raw_dir / f"photo_{len(existing) + 1}{ext}"
    file.save(str(raw_path))

    # ── Step 2: Run face detection (same logic as prepare_dataset.py so i guess no need to use prepare_dataset -this needs confirmation from abdurahim) ──────────
    face_tensor = detect_face(
        image_path=str(raw_path),
        mtcnn=mtcnn,
        check_blur=True,
        blur_threshold=100.0,
    )

    if face_tensor is None:
        raw_path.unlink()  # delete the saved raw image — it's unusable
        return jsonify({"success": False,
                        "message": "No face detected. Please use a clearer photo."}), 400

    # ── Step 3: Save tensor to data/cropped/<name>/ ────────────────────────────
    cropped_dir = PROJECT_ROOT / "data" / "cropped" / name
    cropped_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = cropped_dir / (raw_path.stem + ".pt")
    torch.save(face_tensor, tensor_path)

    return jsonify({
        "success": True,
        "message": f"Face registered for '{name}' successfully!",
        "tensor_shape": list(face_tensor.shape),  # torch.Tensor of shape  [3, 160, 160]
    })


@app.route('/api/fl-results', methods=['GET'])
def fl_results():
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
            return jsonify(data)

    # FL not run yet
    return jsonify({
        "rounds": [], "dp": False,
        "clients": 0,  "model": "Not trained yet",
        "privacy_epsilon": "N/A", "status": "pending"
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)