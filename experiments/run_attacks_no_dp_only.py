"""
run_attacks_no_dp_only.py

Runs model inversion + membership inference on Version A (no DP) ONLY.
This is useful while Version B (DP) is still being fixed.

Usage:
  python experiments/run_attacks_no_dp_only.py

Results:
  - Inversion attack image: results/plots/inversion_no_dp_only.png
  - Membership inference plot: results/plots/membership_no_dp_only.png
  - Membership inference JSON: results/metrics/membership_no_dp_only.json
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (MODEL_NO_DP, PLOTS_DIR, METRICS_DIR,
                    CLIENTS_DIR, ATTACK_ITERATIONS, ATTACK_LR,
                    STYLEGAN_NETWORK_PKL, STYLEGAN_REPO_DIR,
                    STYLEGAN_IDENTITY_W, STYLEGAN_PERCEPTUAL_W,
                    STYLEGAN_LATENT_REG_W)
from src.attacks.model_inversion import (
    load_model_from_checkpoint,
    get_target_embedding,
    run_inversion_attack,
    load_stylegan_generator,
    run_stylegan_inversion_attack
)
from src.attacks.membership_inference import (
    evaluate_membership_inference,
    plot_membership_results
)
import matplotlib.pyplot as plt
import torch

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Check no-DP model exists
    if not os.path.exists(MODEL_NO_DP):
        raise FileNotFoundError(
            f"Model not found: {MODEL_NO_DP}\n"
            "Train the no-DP model first:\n"
            "  python experiments/train_fl_no_dp.py"
        )

    first_client = os.path.join(CLIENTS_DIR, "client_00")

    print("=" * 70)
    print("ATTACK 1: Model Inversion Attack (No DP Baseline)")
    print("=" * 70)
    print(f"Loading model: {MODEL_NO_DP}")
    model_no_dp = load_model_from_checkpoint(MODEL_NO_DP)
    
    print(f"Getting target embedding from: {first_client}")
    target_emb, person_name, original_tensor, target_path = get_target_embedding(model_no_dp, first_client)
    print(f"  Target tensor path: {target_path}")
    
    loss_history_path = os.path.join(METRICS_DIR, "inversion_loss_history_no_dp.json")

    if STYLEGAN_NETWORK_PKL:
        print(f"Running StyleGAN inversion attack ({ATTACK_ITERATIONS} iterations)...")
        print(f"  StyleGAN checkpoint: {STYLEGAN_NETWORK_PKL}")
        if STYLEGAN_REPO_DIR:
            print(f"  StyleGAN repo: {STYLEGAN_REPO_DIR}")

        generator = load_stylegan_generator(
            STYLEGAN_NETWORK_PKL,
            stylegan_repo_dir=STYLEGAN_REPO_DIR or None,
            device="cpu",
        )
        fake_img, final_loss, inversion_stats = run_stylegan_inversion_attack(
            model=model_no_dp,
            target_embedding=target_emb,
            generator=generator,
            iterations=ATTACK_ITERATIONS,
            lr=ATTACK_LR,
            identity_weight=STYLEGAN_IDENTITY_W,
            perceptual_weight=STYLEGAN_PERCEPTUAL_W,
            latent_reg_weight=STYLEGAN_LATENT_REG_W,
            reference_image=original_tensor,
            save_path=None,
            seed=42,
        )
        with open(loss_history_path, "w") as f:
            json.dump(inversion_stats, f, indent=2)
    else:
        print(f"Running inversion attack ({ATTACK_ITERATIONS} iterations)...")
        print(f"  Starting from pure noise (small Gaussian, not from a real face)...")

        fake_img, final_loss = run_inversion_attack(
            model=model_no_dp,
            target_embedding=target_emb,
            iterations=ATTACK_ITERATIONS,
            lr=ATTACK_LR,
            start_tensor=None,  # Use pure noise (no real face as start)
            seed=42,             # deterministic noise for reproducibility
            save_path=None,  # We'll save manually with comparison
            save_loss_path=loss_history_path,
        )

    print(f"Inversion attack complete.")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  (Lower loss = better reconstruction = more privacy leak)")

    # Create comparison figure: original vs reconstructed
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    original_img = (original_tensor.squeeze() + 1) / 2
    axes[0].imshow(original_img.permute(1, 2, 0).numpy())
    axes[0].set_title(f"Original Face ({person_name})")
    axes[0].axis("off")
    
    # Reconstructed image
    reconstructed_img = (fake_img.squeeze() + 1) / 2
    axes[1].imshow(reconstructed_img.permute(1, 2, 0).numpy())
    axes[1].set_title(f"Inversion Attack Result\n(Loss: {final_loss:.4f})")
    axes[1].axis("off")
    
    plt.tight_layout()
    inversion_path = os.path.join(PLOTS_DIR, "inversion_no_dp_only.png")
    plt.savefig(inversion_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {inversion_path}")

    print("\n" + "=" * 70)
    print("ATTACK 2: Membership Inference Attack (No DP Baseline)")
    print("=" * 70)
    print(f"Evaluating membership inference on: {MODEL_NO_DP}")
    
    mi_results = evaluate_membership_inference(
        model=model_no_dp,
        clients_dir=CLIENTS_DIR,
    )
    
    print(f"Membership inference complete.")
    print(f"  Member mean distance: {mi_results['member_mean']:.4f}")
    print(f"  Non-member mean distance: {mi_results['non_member_mean']:.4f}")
    print(f"  Gap: {mi_results['gap']:.4f}")
    print(f"  Advantage: {mi_results['advantage']:.4f}")
    print(f"  (Higher advantage = easier membership inference = more privacy leak)")

    # Save results to JSON
    results_dict = {
        "model": "model_fl_no_dp",
        "inversion_final_loss": final_loss,
        "membership_inference": mi_results,
    }
    
    mi_json_path = os.path.join(METRICS_DIR, "membership_no_dp_only.json")
    with open(mi_json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Saved: {mi_json_path}")

    # Plot membership inference
    wrapped_results = {
        "version_a": mi_results,
        "version_b": mi_results,  # Use same for placeholder; plot will show one bar
    }
    
    mi_plot_path = os.path.join(PLOTS_DIR, "membership_no_dp_only.png")
    plot_membership_results(wrapped_results, mi_plot_path)
    print(f"  Saved: {mi_plot_path}")

    print("\n" + "=" * 70)
    print("SUMMARY: No-DP Baseline Attack Results")
    print("=" * 70)
    print(f"These are baseline measurements WITHOUT Differential Privacy.")
    print(f"Once DP is fixed, we'll compare these to show how much DP helps.")
    print(f"\nResults saved to:")
    print(f"  {inversion_path}")
    print(f"  {mi_plot_path}")
    print(f"  {mi_json_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
