"""
Differential Privacy Training Wrapper using Opacus.

This module wraps PyTorch models, optimizers, and data loaders with Opacus
to enable DP-SGD (Differentially Private Stochastic Gradient Descent) training.

What DP-SGD does in plain terms:
    Before each gradient update, it:
        1. Clips each individual sample's gradient to a max norm (max_grad_norm)
           — so no single student's face data dominates the update
        2. Adds calibrated Gaussian noise to the sum of gradients (noise_multiplier)
           — so the update reveals nothing about any individual sample
    This gives a mathematical privacy guarantee: even if someone gets the model
    weights, they cannot reconstruct what any individual student's face looked like.

Privacy budget (epsilon):
    Every training step "spends" some privacy budget.
    Lower epsilon = stronger privacy but lower accuracy.
    We enforce a hard limit (epsilon_max) — training stops if exceeded.

Owner: Amel
"""

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import torch
import numpy as np
from typing import Tuple, Optional, List
import logging

# Logging setup — outputs [DP] INFO/WARNING/ERROR prefix
logger = logging.getLogger("DP_TRAINING")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[DP] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PrivacyBudgetExceeded(Exception):
    """
    Raised when the privacy budget (epsilon) exceeds the configured limit.

    When this is raised, training must stop immediately. Continuing would
    break the privacy guarantee we promised for this training run.
    """
    pass


class PrivacyMonitor:
    """
    Monitors privacy budget in real-time during training.

    Call check_and_log(privacy_engine) after every epoch.
    It will:
        - Log the current epsilon value
        - Warn when approaching the budget limit (at 80%)
        - Raise PrivacyBudgetExceeded if the limit is crossed
    """

    def __init__(self, epsilon_max: float = 5.0, delta: float = 1e-5):
        """
        Args:
            epsilon_max: Hard limit on privacy budget. Training stops if exceeded.
            delta: The delta in (epsilon, delta)-DP. Keep at 1e-5 unless you know why.
        """
        self.epsilon_max = epsilon_max
        self.delta = delta
        self.epsilon_history: List[float] = []
        self.epoch_counter = 0

    def check_and_log(self, privacy_engine: PrivacyEngine) -> float:
        """
        Check current epsilon and enforce the budget limit.

        Call this after every epoch of training, like this:
            try:
                epsilon = monitor.check_and_log(engine)
            except PrivacyBudgetExceeded:
                break  # stop training

        Args:
            privacy_engine: The PrivacyEngine returned by make_private_with_dp()

        Returns:
            Current epsilon value

        Raises:
            PrivacyBudgetExceeded: If epsilon exceeds epsilon_max
        """
        epsilon = privacy_engine.get_epsilon(delta=self.delta)
        self.epsilon_history.append(epsilon)
        self.epoch_counter += 1

        logger.info(f"Epoch {self.epoch_counter}: epsilon = {epsilon:.4f} / {self.epsilon_max} (delta = {self.delta})")

        # Warn at 80% of budget
        if epsilon > self.epsilon_max * 0.8:
            logger.warning(f"APPROACHING BUDGET: epsilon={epsilon:.4f} is above 80% of limit ({self.epsilon_max * 0.8:.4f})")

        # Hard stop at 100% of budget
        if epsilon > self.epsilon_max:
            logger.error(f"BUDGET EXCEEDED: epsilon={epsilon:.4f} > limit={self.epsilon_max} — stopping training")
            raise PrivacyBudgetExceeded(
                f"Privacy budget exceeded: epsilon={epsilon:.4f} > limit={self.epsilon_max}. "
                f"Training stopped to preserve the privacy guarantee."
            )

        return epsilon

    def get_history(self) -> List[float]:
        """Return the full list of epsilon values recorded so far (one per epoch)."""
        return self.epsilon_history.copy()


def _freeze_backbone_for_dp(model: torch.nn.Module) -> int:
    """
    Freeze all layers except the final linear + BN head of InceptionResnetV1.

    WHY THIS IS NEEDED:
        Opacus (DP-SGD) stores one gradient tensor PER SAMPLE PER TRAINABLE
        PARAMETER. InceptionResnetV1 has ~28M trainable parameters. With 2
        clients running simultaneously, that is 2 × 28M gradient tensors —
        easily exhausting 8-16GB of RAM before training even starts.

        Freezing the backbone reduces trainable params to ~260K (last_linear +
        last_bn only), cutting Opacus memory by ~100x. The backbone already has
        high-quality VGGFace2 pretrained weights, so fine-tuning only the head
        is standard practice in DP federated learning.

    Returns:
        Number of trainable parameters after freezing.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only last_linear and last_bn (the embedding projection head)
    # These live at model.model.last_linear and model.model.last_bn
    inner = getattr(model, "model", model)  # unwrap GlobalFaceModel if needed
    unfrozen_layers = []
    for name in ["last_linear", "last_bn"]:
        layer = getattr(inner, name, None)
        if layer is not None:
            for p in layer.parameters():
                p.requires_grad = True
            unfrozen_layers.append(name)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[DP Backbone Freeze] Frozen all layers. Training only: {unfrozen_layers}. "
        f"Trainable params: {trainable_count:,}"
    )
    return trainable_count


def make_private_with_dp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
    epsilon_max: float = 5.0,
    random_seed: Optional[int] = None,
    batch_memory_manager: bool = False,
    client_id: str = "unknown",
    freeze_backbone: bool = True,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, object, PrivacyEngine, PrivacyMonitor]:
    """
    Wrap a model, optimizer, and data loader with Differential Privacy (DP-SGD).

    This is the main function to call from client.py. It returns everything you
    need to train with DP and monitor the privacy budget each epoch.

    Args:
        model:              PyTorch model to privatize
        optimizer:          Torch optimizer (e.g. Adam)
        train_loader:       DataLoader for training data
        noise_multiplier:   How much Gaussian noise to add to gradients.
                            Higher = more private, lower accuracy.
                            Typical values: 0.5 (weak), 1.1 (strong), 1.5 (very strong)
        max_grad_norm:      Clips each sample's gradient to this L2 norm.
                            Prevents one person's data from dominating the update.
                            Typical value: 1.0
        delta:              The delta in (epsilon, delta)-DP. Keep at 1e-5.
        epsilon_max:        Hard budget limit. Training raises PrivacyBudgetExceeded if crossed.
        random_seed:        Set for reproducible experiments. Use None in production.
        batch_memory_manager: Use memory-efficient batching (useful for large models).
        client_id:          Used in logs to identify which client is training.

    Returns:
        (model_dp, optimizer_dp, loader_dp, privacy_engine, privacy_monitor)

        IMPORTANT: Always use the returned model/optimizer/loader — not the originals.
        IMPORTANT: Call privacy_monitor.check_and_log(privacy_engine) after every epoch.

    Example usage in client.py:
        model_dp, opt_dp, loader_dp, engine, monitor = make_private_with_dp(
            model=self.model,
            optimizer=optimizer,
            train_loader=self.train_loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
            epsilon_max=5.0,
            random_seed=42,
            client_id=self.client_id,
        )
        for epoch in range(LOCAL_EPOCHS):
            train_one_epoch(model_dp, loader_dp, opt_dp)
            try:
                epsilon = monitor.check_and_log(engine)
            except PrivacyBudgetExceeded:
                break
    """

    # FIX: Original code had `if random_seed:` which skips seed=0
    # Corrected to `if random_seed is not None:` so seed=0 works properly
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        logger.info(f"[Reproducibility] Seed set to {random_seed}")

    logger.info(f"[Client-Side] Initializing DP on {client_id}")

    # MEMORY FIX: Freeze backbone before Opacus wraps the model.
    # Opacus allocates per-sample gradient buffers for every trainable param.
    # With a 28M-param model and 2 clients running in parallel this causes OOM.
    # Freezing reduces trainable params to ~260K (head only) — a ~100x reduction.
    if freeze_backbone:
        _freeze_backbone_for_dp(model)

    privacy_engine = PrivacyEngine()

    # Opacus doesn't support BatchNorm. Auto-replace with GroupNorm before calling make_private.
    from opacus.validators import ModuleValidator
    model = ModuleValidator.fix(model)

    # ModuleValidator.fix() may return a new module instance, so rebuild the
    # optimizer against the validated model parameters before wrapping with Opacus.
    optimizer_kwargs = dict(optimizer.defaults)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_private = optimizer.__class__(trainable_params, **optimizer_kwargs)

    model_private, optimizer_private, train_loader_private = privacy_engine.make_private(
        module=model,
        optimizer=optimizer_private,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    if batch_memory_manager:
        train_loader_private = BatchMemoryManager(
            data_loader=train_loader_private,
            max_physical_batch_size=32,
            optimizer=optimizer_private,
        )

    privacy_monitor = PrivacyMonitor(epsilon_max=epsilon_max, delta=delta)

    trainable_count = sum(p.numel() for p in model_private.parameters() if p.requires_grad)
    logger.info(f"[DP Configuration]")
    logger.info(f"  Client:          {client_id}")
    logger.info(f"  noise_multiplier = {noise_multiplier}")
    logger.info(f"  max_grad_norm    = {max_grad_norm}")
    logger.info(f"  target_delta     = {delta}")
    logger.info(f"  epsilon_max      = {epsilon_max}  (hard budget limit)")
    logger.info(f"  random_seed      = {random_seed}")
    logger.info(f"  trainable_params = {trainable_count:,} (backbone frozen={freeze_backbone})")

    return model_private, optimizer_private, train_loader_private, privacy_engine, privacy_monitor


def get_epsilon(privacy_engine: PrivacyEngine, delta: float = 1e-5) -> float:
    """
    Get the current accumulated privacy budget from a PrivacyEngine.

    Epsilon interpretation guide:
        epsilon < 1.0        Very strong privacy
        1.0 to 5.0           Strong privacy (recommended range)
        5.0 to 10.0          Moderate privacy
        epsilon > 10.0       Weak privacy — consider stronger noise

    Args:
        privacy_engine: Returned by make_private_with_dp()
        delta: Must match the delta used when setting up DP

    Returns:
        Current epsilon value
    """
    return privacy_engine.get_epsilon(delta=delta)


# FIX: Removed the dangerous reset_privacy_engine() function that was here.
#
# What it did: privacy_engine.steps = 0  (reset the epsilon counter to zero)
#
# Why it was dangerous: Epsilon accumulates across ALL training rounds in FL.
# Resetting it mid-training would make the system think it had spent epsilon=0
# when it had actually spent epsilon=3 or more. This silently breaks the
# privacy guarantee — you would report "epsilon=3 total" but the real total
# could be much higher. Nobody should call this function during FL training.
#
# If you need it for isolated unit tests, implement it locally in the test file
# with a clear comment explaining why it is safe in that specific context.


class PrivacyConfig:
    """
    Configuration object for DP settings.

    Use the PRIVACY_BUDGETS presets below instead of creating this manually
    unless you have a specific reason to customize.
    """

    def __init__(
        self,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        epsilon_max: float = 5.0,
        random_seed: Optional[int] = None,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.epsilon_max = epsilon_max
        self.random_seed = random_seed

    def __str__(self):
        return (
            f"PrivacyConfig(\n"
            f"  noise_multiplier = {self.noise_multiplier},\n"
            f"  max_grad_norm    = {self.max_grad_norm},\n"
            f"  delta            = {self.delta},\n"
            f"  epsilon_max      = {self.epsilon_max},\n"
            f"  random_seed      = {self.random_seed},\n"
            f")"
        )


# Ready-to-use presets — pass one of these to make_private_with_dp via **vars(config)
# or just read the values out individually.
PRIVACY_BUDGETS = {
    "weak": PrivacyConfig(
        noise_multiplier=0.5,
        max_grad_norm=1.0,
        epsilon_max=10.0,
        random_seed=42,
    ),
    "moderate": PrivacyConfig(
        noise_multiplier=0.8,
        max_grad_norm=1.0,
        epsilon_max=7.0,
        random_seed=42,
    ),
    "strong": PrivacyConfig(
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        epsilon_max=5.0,
        random_seed=42,
    ),
    "very_strong": PrivacyConfig(
        noise_multiplier=1.5,
        max_grad_norm=1.0,
        epsilon_max=3.0,
        random_seed=42,
    ),
}
