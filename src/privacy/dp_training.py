from opacus import PrivacyEngine

def make_private(model, optimizer, data_loader, noise_multiplier, max_grad_norm):
    """
    Wraps the local PyTorch model training inside the Opacus Differential Privacy engine.
    Injects calibrated Gaussian Noise into the gradients to prevent model inversion attacks.
    """
    privacy_engine = PrivacyEngine()
    model_dp, optimizer_dp, loader_dp = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return privacy_engine, model_dp, optimizer_dp, loader_dp

def get_epsilon(privacy_engine, delta):
    """
    AMEL's WORK: Calculates the total mathematical privacy budget (Epsilon) spent so far.
    """
    return privacy_engine.get_epsilon(delta=delta)
