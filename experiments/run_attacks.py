"""
run_attacks.py
Runs model inversion + membership inference on both Version A and Version B.
Must be run AFTER both models are trained:
  python experiments/train_fl_no_dp.py
  python experiments/train_fl_with_dp.py

Usage:
  python experiments/run_attacks.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (MODEL_NO_DP, MODEL_WITH_DP, PLOTS_DIR, METRICS_DIR,
                    CLIENTS_DIR, ATTACK_ITERATIONS, ATTACK_LR)
from src.attacks.model_inversion import attack_both_models, load_model_from_checkpoint
from src.attacks.membership_inference import compare_membership_inference, plot_membership_results

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Check both models exist
    for path in [MODEL_NO_DP, MODEL_WITH_DP]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Run training scripts first:\n"
                "  python experiments/train_fl_no_dp.py\n"
                "  python experiments/train_fl_with_dp.py"
            )

    # Pick first client as attack target (use their photos as ground truth)
    first_client = os.path.join(CLIENTS_DIR, "client_00")

    print("=" * 60)
    print("ATTACK 1: Model Inversion Attack")
    print("=" * 60)
    inversion_results = attack_both_models(
        model_no_dp_path=MODEL_NO_DP,
        model_with_dp_path=MODEL_WITH_DP,
        client_dir=first_client,
        output_dir=PLOTS_DIR,
        iterations=ATTACK_ITERATIONS,
    )
    print(f"Inversion attack complete.")
    print(f"  Version A final loss: {inversion_results['no_dp_final_loss']:.4f}")
    print(f"  Version B final loss: {inversion_results['with_dp_final_loss']:.4f}")

    print("\n" + "=" * 60)
    print("ATTACK 2: Membership Inference Attack")
    print("=" * 60)
    model_no_dp   = load_model_from_checkpoint(MODEL_NO_DP)
    model_with_dp = load_model_from_checkpoint(MODEL_WITH_DP)

    mi_results = compare_membership_inference(
        model_no_dp=model_no_dp,
        model_with_dp=model_with_dp,
        clients_dir=CLIENTS_DIR,
        output_path=os.path.join(METRICS_DIR, "membership_inference.json"),
    )
    plot_membership_results(
        results=mi_results,
        output_path=os.path.join(PLOTS_DIR, "membership_inference.png"),
    )
    print(f"Membership inference complete.")
    print(f"  Version A advantage: {mi_results['version_a']['advantage']:.4f}")
    print(f"  Version B advantage: {mi_results['version_b']['advantage']:.4f}")

    print("\n" + "=" * 60)
    print("All attacks complete. Results saved to:")
    print(f"  {PLOTS_DIR}/attack_no_dp.png")
    print(f"  {PLOTS_DIR}/attack_with_dp.png")
    print(f"  {PLOTS_DIR}/attack_comparison.png")
    print(f"  {PLOTS_DIR}/membership_inference.png")
    print(f"  {METRICS_DIR}/membership_inference.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
