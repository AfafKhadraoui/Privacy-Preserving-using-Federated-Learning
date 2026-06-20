"""
Secure Aggregation for Federated Learning.

What SecAgg does in plain terms:
    Normally, the server sees each client's individual model update.
    SecAgg prevents this — clients mask their updates with random numbers
    that cancel out when summed, so the server gets the correct aggregate
    but cannot inspect any individual client's contribution.

    This matters for our face recognition project because model updates
    can leak information about the training data (students' faces).
    SecAgg adds a layer of protection even against a compromised server.

Current status (important — read before using):
    The installed Flower version uses the NEW workflow/mod API for SecAgg.
    The old strategy-based SecAggPlusStrategy class no longer exists.

    This means:
        get_secagg_strategy() → falls back to plain FedAvg (SecAgg NOT active)
        get_secagg_workflow()  → returns the correct workflow object (SecAgg ready)
        get_secagg_client_mod() → returns the correct client mod (SecAgg ready)

    To fully activate SecAgg, the project needs to migrate from:
        fl.server.start_server(strategy=...) + fl.client.start_numpy_client(...)
    to:
        ServerApp + ClientApp (Flower's newer launch system)

    This migration is the known next step for the project.
    Until then, DP + signing + encryption are the active privacy layers.

Owner: Amel
"""

import flwr as fl
from typing import Any, Optional, Type


def has_legacy_secagg_strategy() -> bool:
    """
    Check if the old strategy-based SecAgg API is available.

    Returns True only on older Flower versions (before ~1.29).
    Expected to return False on current installations.
    """
    return hasattr(fl.server.strategy, "SecAggPlusStrategy")


def has_workflow_secagg() -> bool:
    """
    Check if the new workflow/mod SecAgg API is available.

    This is the current Flower API. Returns True on Flower >= 1.29.
    Having this available means SecAgg CAN be activated,
    but only after the ServerApp/ClientApp migration.
    """
    has_server_workflow = (
        hasattr(fl.server, "workflow")
        and hasattr(fl.server.workflow, "SecAggPlusWorkflow")
    )
    has_client_mod = (
        hasattr(fl.client, "mod")
        and hasattr(fl.client.mod, "secaggplus_mod")
    )
    return has_server_workflow and has_client_mod


def print_secagg_status():
    """
    Print a clear summary of what SecAgg API is available and what is active.

    Call this at server startup so the team always knows the current state.
    """
    if has_legacy_secagg_strategy():
        print("[SecAgg] Status: Legacy SecAggPlusStrategy available — strategy wrapping ACTIVE")
    elif has_workflow_secagg():
        print("[SecAgg] Status: Workflow/mod API available (Flower >= 1.29)")
        print("[SecAgg] Status: SecAgg is NOT active in the current run")
        print("[SecAgg] Status: Reason: requires ServerApp/ClientApp migration (known next step)")
        print("[SecAgg] Status: Active privacy layers: DP + Ed25519 signing + AES-256-GCM encryption")
    else:
        print("[SecAgg] Status: No SecAgg API detected in installed Flower version")
        print("[SecAgg] Status: Running without secure aggregation")


def get_secagg_strategy(
    base_strategy_class: Type[fl.server.strategy.FedAvg] = fl.server.strategy.FedAvg,
    use_dp: bool = False,
    **kwargs,
):
    """
    Attempt to wrap a Flower strategy with legacy SecAgg.

    On current Flower versions this will fall back to the base strategy
    because SecAggPlusStrategy no longer exists. This is expected and
    the fallback is intentional — see module docstring for context.

    Args:
        base_strategy_class: The strategy class to instantiate (e.g. SaveModelStrategy)
        use_dp: Passed to the strategy constructor if it accepts it
        **kwargs: Additional arguments forwarded to the strategy constructor

    Returns:
        SecAggPlusStrategy wrapping base_strategy if legacy API available,
        otherwise just base_strategy with a clear warning printed.
    """
    # Instantiate the base strategy — try with use_dp first, fall back without
    try:
        base_strategy = base_strategy_class(use_dp=use_dp, **kwargs)
    except TypeError:
        base_strategy = base_strategy_class(**kwargs)

    secagg_cls = getattr(fl.server.strategy, "SecAggPlusStrategy", None)

    if secagg_cls is None:
        # This is the expected case on current Flower — not an error
        print("[SecAgg] Legacy SecAggPlusStrategy not available — using base strategy")
        print("[SecAgg] SecAgg will be activated after ServerApp/ClientApp migration")
        return base_strategy

    # Legacy path — wraps with SecAgg if the old API is available
    print("[SecAgg] Wrapping strategy with SecAggPlusStrategy (legacy API)")
    print("[SecAgg] Individual client updates will be encrypted and masked")
    return secagg_cls(base_strategy=base_strategy)


def get_secagg_workflow(
    num_shares: float = 1.0,
    reconstruction_threshold: float = 0.5,
    max_weight: float = 1000.0,
    clipping_range: float = 8.0,
    quantization_range: int = 2 ** 22,
    modulus_range: int = 2 ** 32,
    timeout: Optional[float] = None,
) -> Any:
    """
    Build a SecAggPlusWorkflow for use with the new Flower ServerApp API.

    This is the CORRECT way to use SecAgg in current Flower versions.
    It requires ServerApp/ClientApp — see migration guide in module docstring.

    Args:
        num_shares:               Number of shares each client sends (1.0 = all clients)
        reconstruction_threshold: Minimum fraction of shares needed to reconstruct (0.5 = 50%)
        max_weight:               Maximum value any weight can have before quantization
        clipping_range:           Range for weight clipping before quantization
        quantization_range:       Number of quantization levels
        modulus_range:            Modulus for the secret sharing arithmetic
        timeout:                  Seconds to wait for clients before timing out (None = no limit)

    Returns:
        SecAggPlusWorkflow instance if available, None otherwise

    Example (after ServerApp migration):
        workflow = get_secagg_workflow(num_shares=1.0, reconstruction_threshold=0.5)
        app = ServerApp(config=ServerConfig(num_rounds=3), workflow=workflow)
    """
    workflow_cls = getattr(getattr(fl.server, "workflow", object()), "SecAggPlusWorkflow", None)

    if workflow_cls is None:
        print("[SecAgg] WARNING: SecAggPlusWorkflow not available in this Flower version")
        return None

    return workflow_cls(
        num_shares=num_shares,
        reconstruction_threshold=reconstruction_threshold,
        max_weight=max_weight,
        clipping_range=clipping_range,
        quantization_range=quantization_range,
        modulus_range=modulus_range,
        timeout=timeout,
    )


def get_secagg_client_mod(use_plus: bool = True) -> Any:
    """
    Return the client-side mod needed for workflow-based SecAgg.

    Must be used together with get_secagg_workflow() on the server side.
    Both server workflow AND client mod must be active for SecAgg to work.

    Args:
        use_plus: Use SecAggPlus (recommended). Falls back to basic SecAgg if not available.

    Returns:
        The mod object to pass to ClientApp, or None if not available.

    Example (after ClientApp migration):
        mod = get_secagg_client_mod()
        app = ClientApp(client_fn=lambda cid: MyClient(cid), mods=[mod])
    """
    mod_namespace = getattr(fl.client, "mod", None)

    if mod_namespace is None:
        print("[SecAgg] WARNING: Flower client mod namespace not available")
        return None

    if use_plus and hasattr(mod_namespace, "secaggplus_mod"):
        return mod_namespace.secaggplus_mod

    if hasattr(mod_namespace, "secagg_mod"):
        return mod_namespace.secagg_mod

    print("[SecAgg] WARNING: No secure aggregation client mod found")
    return None


class SecureAggregationConfig:
    """
    Configuration for secure aggregation settings.

    Use RECOMMENDED_CONFIG below unless you have a reason to customize.
    """

    def __init__(
        self,
        enabled: bool = True,
        protocol_version: str = "SecAggPlus",
        timeout_seconds: int = 600,
        minimum_threshold: Optional[int] = None,
    ):
        """
        Args:
            enabled:           Whether SecAgg should be used (still requires migration to activate)
            protocol_version:  "SecAggPlus" (Bonawitz et al., 2016) — only supported option
            timeout_seconds:   How long to wait for all clients before aborting a round
            minimum_threshold: Minimum number of clients required for aggregation.
                               None means all clients must participate.
        """
        self.enabled = enabled
        self.protocol_version = protocol_version
        self.timeout_seconds = timeout_seconds
        self.minimum_threshold = minimum_threshold

    def __str__(self):
        return (
            f"SecureAggregationConfig(\n"
            f"  enabled           = {self.enabled},\n"
            f"  protocol_version  = {self.protocol_version},\n"
            f"  timeout_seconds   = {self.timeout_seconds},\n"
            f"  minimum_threshold = {self.minimum_threshold},\n"
            f")"
        )


# Recommended configuration for this project
RECOMMENDED_CONFIG = SecureAggregationConfig(
    enabled=True,
    protocol_version="SecAggPlus",
    timeout_seconds=300,
    minimum_threshold=None,  # All clients must participate
)
