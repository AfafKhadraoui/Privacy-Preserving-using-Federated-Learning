"""
Privacy Accounting — Tracking and Reporting Privacy Budgets Across FL Rounds.

What this module does:
    After each FL round, the client knows how much privacy budget (epsilon) was spent
    and what accuracy the model achieved. This module records that history and
    generates a report + plot at the end of training.

    The privacy-accuracy tradeoff chart is useful for your presentation:
    it visually shows that higher privacy (lower epsilon) comes at the cost
    of some model accuracy — this is the fundamental tradeoff in DP.

How to use it (already wired into client.py):
    accountant = PrivacyAccountant(delta=1e-5)

    # After each evaluate() call:
    accountant.log_round(round_number=1, epsilon=3.36, accuracy=0.87, loss=0.12)

    # At the end of training:
    accountant.save_report("results/metrics/privacy_accounting_client_00.json")
    accountant.plot_tradeoff("results/plots/privacy_tradeoff_client_00.png")

Owner: Amel
"""

import json
import os
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt


class PrivacyAccountant:
    """
    Records privacy budget consumption and model accuracy per FL round.

    One instance per client — created in FaceClient.__init__() and used
    in evaluate() to log each round's epsilon and accuracy.
    """

    def __init__(self, delta: float = 1e-5):
        """
        Args:
            delta: The delta in (epsilon, delta)-DP. Should match what was used in dp_training.py.
        """
        self.delta = delta
        self.rounds_history: List[Dict] = []

    def log_round(
        self,
        round_number: int,
        epsilon: float,
        accuracy: float,
        loss: Optional[float] = None,
    ):
        """
        Record the privacy budget and model performance for one FL round.

        Args:
            round_number: Which FL round this is (1, 2, 3, ...)
            epsilon:      Privacy budget spent so far (cumulative, from PrivacyMonitor)
            accuracy:     Model accuracy on local data this round (0.0 to 1.0)
            loss:         Training loss this round (optional but useful for plotting)
        """
        entry = {
            "round": round_number,
            "epsilon": epsilon,
            "accuracy": accuracy,
            "loss": loss,
        }
        self.rounds_history.append(entry)
        print(f"[Privacy Accounting] Round {round_number}: epsilon={epsilon:.4f}, accuracy={accuracy:.4f}")

    def save_report(self, output_path: str):
        """
        Save the full privacy accounting history to a JSON file.

        The JSON includes delta, all rounds, and the final epsilon value.
        This is useful to attach to your project report or share with the team.

        Args:
            output_path: Where to save the file e.g. results/metrics/privacy_accounting_client_00.json
        """
        # FIX: Was crashing if rounds_history was empty — now handles gracefully
        final_epsilon = self.rounds_history[-1]["epsilon"] if self.rounds_history else None

        report = {
            "delta": self.delta,
            "total_rounds": len(self.rounds_history),
            "final_epsilon": final_epsilon,
            "rounds": self.rounds_history,
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[Privacy Accounting] Report saved to {output_path}")

    def get_privacy_accuracy_tradeoff(self) -> Tuple[List[float], List[float]]:
        """
        Return (epsilons, accuracies) lists for use in plotting.

        Each value corresponds to one FL round in order.
        """
        epsilons = [r["epsilon"] for r in self.rounds_history]
        accuracies = [r["accuracy"] for r in self.rounds_history]
        return epsilons, accuracies

    def plot_tradeoff(self, output_path: str = "results/plots/privacy_accuracy_tradeoff.png"):
        """
        Generate and save the privacy-accuracy tradeoff chart.

        The chart shows how epsilon grew over rounds and how accuracy changed.
        Good for presentations: it visually demonstrates the DP tradeoff.

        Args:
            output_path: Where to save the PNG image
        """
        # FIX: Was crashing on empty history with unhelpful IndexError
        if not self.rounds_history:
            print("[Privacy Accounting] WARNING: No rounds recorded — skipping plot")
            return

        rounds = [r["round"] for r in self.rounds_history]
        epsilons = [r["epsilon"] for r in self.rounds_history]
        accuracies = [r["accuracy"] for r in self.rounds_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left chart: epsilon growing over rounds (privacy cost over time)
        ax1.plot(rounds, epsilons, "o-", linewidth=2, markersize=8, color="steelblue")
        ax1.set_xlabel("FL Round", fontsize=12)
        ax1.set_ylabel("Cumulative Privacy Budget (epsilon)", fontsize=12)
        ax1.set_title("Privacy Budget Consumption Over Rounds", fontsize=13)
        ax1.grid(True, alpha=0.3)

        # Right chart: epsilon vs accuracy (the tradeoff)
        ax2.plot(epsilons, accuracies, "s-", linewidth=2, markersize=8, color="darkorange")
        ax2.set_xlabel("Privacy Budget (epsilon) — lower is more private", fontsize=12)
        ax2.set_ylabel("Model Accuracy", fontsize=12)
        ax2.set_title("Privacy-Accuracy Tradeoff", fontsize=13)
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"DP Training Summary (delta={self.delta})", fontsize=14, y=1.02)
        plt.tight_layout()

        # FIX: Original code crashed when output_path had no directory component
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[Privacy Accounting] Plot saved to {output_path}")
        plt.close()


class PrivacyBudgetValidator:
    """
    Simple utility to check whether a given epsilon value is within budget.

    Used as a quick sanity check — the hard enforcement is done by
    PrivacyMonitor in dp_training.py. This is for reporting/alerting.
    """

    def __init__(self, epsilon_budget: float = 5.0, delta: float = 1e-5):
        """
        Args:
            epsilon_budget: Maximum acceptable epsilon. Lower = stricter privacy.
            delta:          The delta in (epsilon, delta)-DP.
        """
        self.epsilon_budget = epsilon_budget
        self.delta = delta

    def check_budget(self, epsilon: float) -> bool:
        """
        Check whether an epsilon value is within budget.

        Args:
            epsilon: The epsilon value to check

        Returns:
            True if within budget, False if exceeded
        """
        if epsilon > self.epsilon_budget:
            print(f"[Privacy Validator] ALERT: epsilon={epsilon:.4f} EXCEEDS budget of {self.epsilon_budget}")
            return False

        remaining = self.epsilon_budget - epsilon
        print(f"[Privacy Validator] OK: epsilon={epsilon:.4f} within budget={self.epsilon_budget} (remaining={remaining:.4f})")
        return True

    def __str__(self):
        return f"PrivacyBudgetValidator(epsilon_max={self.epsilon_budget}, delta={self.delta})"


class PrivacySweepExperiment:
    """
    Run and record a privacy-utility sweep across different noise levels.

    A sweep experiment trains the same model multiple times with different
    noise_multiplier values to find the best privacy-accuracy tradeoff
    for your specific dataset and model.

    Usage:
        sweep = PrivacySweepExperiment(output_dir="results/privacy_sweep")
        for sigma in [0.5, 0.8, 1.1, 1.5]:
            # ... run training with this sigma ...
            sweep.add_result(sigma=sigma, epsilon=final_epsilon, accuracy=final_accuracy)
        sweep.save_results()
        sweep.plot_tradeoff()
    """

    def __init__(self, output_dir: str = "results/privacy_sweep"):
        self.output_dir = output_dir
        self.results: Dict[float, Dict] = {}
        os.makedirs(output_dir, exist_ok=True)

    def add_result(self, sigma: float, epsilon: float, accuracy: float):
        """
        Record the result for one noise multiplier value.

        Args:
            sigma:    The noise_multiplier used in this run
            epsilon:  Final epsilon achieved
            accuracy: Final model accuracy achieved
        """
        self.results[sigma] = {
            "sigma": sigma,
            "epsilon": epsilon,
            "accuracy": accuracy,
        }
        print(f"[Privacy Sweep] sigma={sigma}: epsilon={epsilon:.4f}, accuracy={accuracy:.4f}")

    def save_results(self):
        """Save all sweep results to JSON."""
        results_path = os.path.join(self.output_dir, "sweep_results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[Privacy Sweep] Results saved to {results_path}")

    def plot_tradeoff(self):
        """
        Plot accuracy vs noise level and accuracy vs epsilon side by side.

        FIX: Added early return for empty results to avoid crash.
        """
        if not self.results:
            print("[Privacy Sweep] WARNING: No results recorded — skipping plot")
            return

        sigmas = sorted(self.results.keys())
        accuracies = [self.results[s]["accuracy"] for s in sigmas]
        epsilons = [self.results[s]["epsilon"] for s in sigmas]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(sigmas, accuracies, "o-", linewidth=2, markersize=8)
        ax1.set_xlabel("Noise Multiplier (sigma)", fontsize=12)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.set_title("Accuracy vs Noise Level", fontsize=14)
        ax1.grid(True, alpha=0.3)

        ax2.plot(epsilons, accuracies, "s-", linewidth=2, markersize=8, color="orange")
        ax2.set_xlabel("Privacy Budget (epsilon)", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title("Privacy-Accuracy Tradeoff", fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "privacy_sweep_plot.png")
        plt.savefig(plot_path, dpi=300)
        print(f"[Privacy Sweep] Plot saved to {plot_path}")
        plt.close()


# Default validator with standard settings — import and use directly if needed
DEFAULT_VALIDATOR = PrivacyBudgetValidator(epsilon_budget=5.0, delta=1e-5)
