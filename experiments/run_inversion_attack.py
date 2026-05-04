#!/usr/bin/env python
"""
run_inversion_attack.py
Runs the model inversion attack on one or more raw target images.
- MTCNN detects and aligns each face before embedding extraction.
- Runs BOTH MobileStyleGAN and Pixel-Space attacks on each target.
- Produces a 5-panel comparison plot per target.

Usage:
    python experiments/run_inversion_attack.py
"""
import sys
import os

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import MODEL_NO_DP, MODEL_WITH_DP, PLOTS_DIR, CLIENTS_DIR, ATTACK_ITERATIONS, ATTACK_LR
from src.attacks.model_inversion import attack_both_models

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Target images (raw photos — MTCNN will crop the face automatically) ──────
RAW_DIR = os.path.join(project_root, "data", "raw")
TARGET_IMAGES = [
    os.path.join(RAW_DIR, "Abdelmadjid_Tebboune.jpg"),
    os.path.join(RAW_DIR, "Taher_Elgamal.jpg"),
]

# Fallback client dir (used only if an image file is missing)
client_dir = os.path.join(CLIENTS_DIR, "client_00")

print("=" * 60)
print(f"Model Inversion Attack  —  {ATTACK_ITERATIONS} iterations")
print("=" * 60)

all_results = []

for image_path in TARGET_IMAGES:
    tag = os.path.splitext(os.path.basename(image_path))[0]  # e.g. "Abdelmadjid_Tebboune"

    if not os.path.exists(image_path):
        print(f"\n[SKIP] File not found: {image_path}")
        continue

    print(f"\n>>> Target: {tag}")
    print("-" * 60)

    res = attack_both_models(
        model_no_dp_path   = MODEL_NO_DP,
        model_with_dp_path = MODEL_WITH_DP,
        client_dir         = client_dir,
        output_dir         = PLOTS_DIR,
        iterations         = ATTACK_ITERATIONS,
        attack_lr          = ATTACK_LR,
        plot_file_tag      = tag,
        target_image_path  = image_path,
    )
    all_results.append(res)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅  All attacks complete!")
print("=" * 60)
for res in all_results:
    tag = res["client_tag"]
    print(f"\n  [{tag}]")
    if res.get("msg_no_dp_loss") is not None:
        print(f"    MobileStyleGAN  No-DP  loss : {res['msg_no_dp_loss']:.4f}")
        print(f"    MobileStyleGAN  With-DP loss : {res['msg_dp_loss']:.4f}")
    print(f"    Pixel-Space     No-DP  loss : {res['pix_no_dp_loss']:.4f}")
    print(f"    Pixel-Space     With-DP loss : {res['pix_dp_loss']:.4f}")

print(f"\nPlots saved to: {PLOTS_DIR}")
