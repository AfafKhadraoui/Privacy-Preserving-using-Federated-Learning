#!/usr/bin/env python
"""Quick script to run single-client inversion attack."""
import sys
import os

# Add project root to sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from config import MODEL_NO_DP, MODEL_WITH_DP, PLOTS_DIR
from src.attacks.model_inversion import attack_both_models

os.makedirs(PLOTS_DIR, exist_ok=True)

client_dir = os.path.join(_ROOT, "data", "clients", "client_00")

print("=" * 60)
print("Running inversion attack on client_00...")
print("=" * 60)

res = attack_both_models(
    model_no_dp_path=MODEL_NO_DP,
    model_with_dp_path=MODEL_WITH_DP,
    client_dir=client_dir,
    output_dir=PLOTS_DIR,
    iterations=300,
    attack_lr=0.01,
    plot_file_tag="client_00",
)

print("\nV Attack complete!")
print("=" * 60)
print(f"Client: {res['client_tag']}")
print(f"No-DP final loss: {res['no_dp_final_loss']:.4f}")
print(f"With-DP final loss: {res['with_dp_final_loss']:.4f}")
print(f"\nPlots saved to: {PLOTS_DIR}")
