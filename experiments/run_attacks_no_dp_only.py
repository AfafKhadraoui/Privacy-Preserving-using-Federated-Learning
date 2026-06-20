"""
run_attacks_no_dp_only.py

Runs embedding-only model inversion on Version A (no DP) ONLY.

Usage:
  python experiments/run_attacks_no_dp_only.py
  python experiments/run_attacks_no_dp_only.py --client client_01

Results:
  - Inversion plot: results/plots/inversion_no_dp_only_<client>.png
  - Summary JSON: results/metrics/inversion_no_dp_only_<client>.json
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_NO_DP,
    PLOTS_DIR,
    METRICS_DIR,
    CLIENTS_DIR,
    CROPPED_DIR,
    ATTACK_ITERATIONS,
    ATTACK_LR,
    STYLEGAN_NETWORK_PKL,
    STYLEGAN_REPO_DIR,
    STYLEGAN_IDENTITY_W,
    STYLEGAN_LATENT_REG_W,
)
from src.attacks.model_inversion import (
    load_model_from_checkpoint,
    whitebox_target_embedding_from_crop,
    evaluator_ground_truth_crop,
    run_inversion_attack,
    load_stylegan_generator,
    run_stylegan_inversion_attack,
)
import matplotlib.pyplot as plt
import torch


def resolve_client_dir(clients_root: str, cid_raw: str) -> str:
    s = cid_raw.strip()
    if s.startswith("client_"):
        name = s
    elif s.isdigit():
        name = f"client_{int(s):02d}"
    else:
        name = f"client_{s}"
    return os.path.join(clients_root, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=str, default="client_00", help="e.g. client_00 / client_01")
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    client_dir = resolve_client_dir(CLIENTS_DIR, args.client)
    tag = os.path.basename(client_dir)
    vic_abs = os.path.abspath(client_dir)

    if not os.path.exists(MODEL_NO_DP):
        raise FileNotFoundError(
            f"Model not found: {MODEL_NO_DP}\nTrain: python experiments/train_fl_no_dp.py"
        )
    if not os.path.isdir(client_dir):
        raise FileNotFoundError(f"No client folder: {client_dir}")

    print("=" * 70)
    print("Model inversion (No DP baseline, embedding-only)")
    print("=" * 70)
    print(f"Target embeddings from: {client_dir}")

    model_no_dp = load_model_from_checkpoint(MODEL_NO_DP)
    target_emb, person_name = whitebox_target_embedding_from_crop(model_no_dp, client_dir)

    prior_exclude_roots = [vic_abs]
    if os.path.isdir(CROPPED_DIR) and person_name:
        cap = os.path.abspath(os.path.join(CROPPED_DIR, person_name))
        if os.path.isdir(cap):
            prior_exclude_roots.append(cap)

    loss_history_path = os.path.join(METRICS_DIR, f"inversion_loss_history_no_dp_{tag}.json")

    if STYLEGAN_NETWORK_PKL:
        print(f"StyleGAN inversion ({ATTACK_ITERATIONS} iters)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = load_stylegan_generator(
            STYLEGAN_NETWORK_PKL,
            stylegan_repo_dir=STYLEGAN_REPO_DIR or None,
            device=device,
        )
        fake_img, final_loss, inversion_stats = run_stylegan_inversion_attack(
            model=model_no_dp,
            target_embedding=target_emb,
            generator=generator,
            iterations=ATTACK_ITERATIONS,
            lr=ATTACK_LR,
            identity_weight=STYLEGAN_IDENTITY_W,
            latent_reg_weight=STYLEGAN_LATENT_REG_W,
            save_path=None,
            seed=42,
        )
        with open(loss_history_path, "w") as f:
            json.dump(inversion_stats, f, indent=2)
    else:
        prior = CROPPED_DIR if os.path.isdir(CROPPED_DIR) else None
        print(f"Pixel inversion ({ATTACK_ITERATIONS} iters)...")
        fake_img, final_loss = run_inversion_attack(
            model=model_no_dp,
            target_embedding=target_emb,
            iterations=ATTACK_ITERATIONS,
            lr=ATTACK_LR,
            start_tensor=None,
            prior_dir=prior,
            prior_exclude_prefixes=prior_exclude_roots,
            save_path=None,
            save_loss_path=loss_history_path,
            seed=42,
        )

    print(f"Done. Loss: {final_loss:.4f}")

    gt_tensor, plot_name = evaluator_ground_truth_crop(client_dir)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    gt_img = ((gt_tensor.squeeze() + 1) / 2).clamp(0, 1)
    axes[0].imshow(gt_img.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title(f"GT (evaluator only)\n({plot_name} / {person_name})")
    axes[0].axis("off")

    reconstructed = ((fake_img.squeeze() + 1) / 2).clamp(0, 1)
    axes[1].imshow(reconstructed.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title(f"Inversion\nloss={final_loss:.4f}")
    axes[1].axis("off")

    plt.tight_layout()
    inversion_path = os.path.join(PLOTS_DIR, f"inversion_no_dp_only_{tag}.png")
    plt.savefig(inversion_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {inversion_path}")

    summary_path = os.path.join(METRICS_DIR, f"inversion_no_dp_only_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump({"model": "model_fl_no_dp", "client": tag, "inversion_final_loss": final_loss}, f, indent=2)
    print(f"Saved {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
