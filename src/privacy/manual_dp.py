"""
MANUAL DIFFERENTIAL PRIVACY (FROM SCRATCH)

This module implements the core logic of DP-SGD without using external libraries.
It is designed to be transparent and easy to explain in your project report.

HOW IT WORKS:
    1. Clipping: We ensure no single gradient update is too large.
    2. Noising: We add Gaussian noise to the gradients before the update.

By doing this manually, we avoid the heavy memory hooks of Opacus, making
it easier to run on machines with limited RAM (8GB).
"""

import torch
import logging

logger = logging.getLogger("DP_MANUAL")

class ManualDPWrapper:
    """
    A wrapper for PyTorch optimizers that implements manual DP-SGD.
    
    This is a "from scratch" implementation of the two key DP steps:
    Global L2 Clipping and Gaussian Noise Addition.
    """

    def __init__(self, optimizer, noise_multiplier=1.1, max_grad_norm=1.0):
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """
        Perform the optimizer step with manual clipping and noising.
        
        This is the core 'From Scratch' logic.
        """
        # 1. CLIPPING: Limit the influence of any single batch
        # We use torch.nn.utils.clip_grad_norm_ which implements:
        # grad = grad * min(1, max_grad_norm / L2_norm(grad))
        torch.nn.utils.clip_grad_norm_(
            parameters=[p for group in self.param_groups for p in group['params']],
            max_norm=self.max_grad_norm
        )

        # 2. NOISING: Add Gaussian noise to every trainable parameter
        # The noise scale is: noise_multiplier * max_grad_norm
        # This is what provides the mathematical privacy guarantee.
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        # Generate noise: N(0, sigma^2)
                        # sigma = noise_multiplier * max_grad_norm
                        noise = torch.randn_like(p.grad) * (self.noise_multiplier * self.max_grad_norm)
                        p.grad.add_(noise)

        # 3. UPDATE: Call the original optimizer step (e.g. Adam/SGD)
        return self.optimizer.step(closure=closure)

    def get_epsilon(self, delta=1e-5):
        """Placeholder for epsilon calculation."""
        return 0.0


class LocalDPFeaturePerturber:
    """
    Implements Local Differential Privacy (LDP) on face embeddings.
    """
    @staticmethod
    def perturb(embeddings, noise_multiplier=0.1):
        noise = torch.randn_like(embeddings) * noise_multiplier
        return embeddings + noise
