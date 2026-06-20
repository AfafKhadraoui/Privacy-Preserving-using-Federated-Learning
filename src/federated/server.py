"""
FL Server — Collects client updates, validates signatures, aggregates, saves model.

Security layers active here:
    1. Signature validation  — rejects unsigned or tampered updates before aggregation
    2. Protocol version check — rejects clients running old/wrong protocol
    3. Audit logging         — every round and security event is logged with hash chain
    4. SecAgg (best-effort)  — wraps strategy if available in this Flower version

NOTE on SecAgg: The current Flower version uses workflow/mod API for SecAgg, not the
legacy strategy wrapper. This means get_secagg_strategy() will fall back to plain
FedAvg with a clear warning. Full SecAgg activation requires ServerApp/ClientApp
migration — this is the known next step.
"""

import os
import argparse
import sys
import base64
import flwr as fl
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.model.face_model import get_model, set_parameters, save_model
from src.privacy.signing import SignatureValidator
from src.privacy.audit_logging import get_audit_log


# One shared audit log for the server — logs all rounds and security events
_server_audit_log = None

def get_server_audit_log():
    global _server_audit_log
    if _server_audit_log is None:
        log_path = os.path.join(config.METRICS_DIR, "audit_server.log")
        _server_audit_log = get_audit_log(log_path)
    return _server_audit_log


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, use_dp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dp = use_dp
        self.signature_validator = SignatureValidator()
        self.audit_log = get_server_audit_log()

        # Log server startup
        self.audit_log.log_event(
            event_type="server_started",
            details={
                "dp_enabled": use_dp,
                "num_rounds": config.NUM_ROUNDS,
                "min_clients": config.MIN_CLIENTS,
                "enforce_signatures": config.ENFORCE_SIGNATURES,
                "protocol_version": config.PROTOCOL_VERSION,
            },
            severity="INFO"
        )

    def _validate_and_filter_results(self, server_round, results):
        """
        Validate signed metadata for each client update before aggregation.

        For each update we check:
            1. Signature metadata is present (client_id, signature, model_hash)
            2. Protocol version matches server config
            3. Public key exists for this client
            4. Base64 signature decodes correctly
            5. Ed25519 signature is valid for the claimed manifest

        Any update failing these checks is rejected and logged as a security alert.
        If ALL updates are rejected, aggregation is skipped for this round.
        """
        if not config.ENFORCE_SIGNATURES:
            # Log that we are running without signature enforcement — useful to know
            self.audit_log.log_event(
                event_type="signature_enforcement_disabled",
                details={"round": server_round},
                severity="WARNING"
            )
            return results

        filtered = []
        rejected_count = 0

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            client_id = metrics.get("client_id")
            signature_b64 = metrics.get("signature_b64")
            model_hash = metrics.get("model_hash")
            protocol_version = metrics.get("protocol_version")
            round_id = metrics.get("round_id", server_round)

            # Check 1: required fields present
            if not client_id or not signature_b64 or not model_hash:
                print(f"[Signing] Rejecting update from {client_proxy.cid}: missing signature metadata")
                self.audit_log.log_security_alert(
                    alert_type="missing_signature_metadata",
                    details={"round": server_round, "cid": str(client_proxy.cid)}
                )
                rejected_count += 1
                continue

            # Check 2: protocol version matches
            if protocol_version != config.PROTOCOL_VERSION:
                print(f"[Signing] Rejecting {client_id}: protocol mismatch ({protocol_version} != {config.PROTOCOL_VERSION})")
                self.audit_log.log_security_alert(
                    alert_type="protocol_version_mismatch",
                    details={
                        "round": server_round,
                        "client_id": client_id,
                        "client_version": protocol_version,
                        "server_version": config.PROTOCOL_VERSION,
                    }
                )
                rejected_count += 1
                continue

            # Check 3: public key exists
            public_key_path = os.path.join(config.KEYS_DIR, f"{client_id}_public.pem")
            if client_id not in self.signature_validator.public_keys:
                if not os.path.exists(public_key_path):
                    print(f"[Signing] Rejecting {client_id}: public key not found at {public_key_path}")
                    self.audit_log.log_security_alert(
                        alert_type="public_key_not_found",
                        details={"round": server_round, "client_id": client_id, "path": public_key_path}
                    )
                    rejected_count += 1
                    continue
                self.signature_validator.register_client_key(client_id, public_key_path)

            # Check 4: base64 decode
            try:
                signature = base64.b64decode(signature_b64)
            except Exception:
                print(f"[Signing] Rejecting {client_id}: invalid base64 signature")
                self.audit_log.log_security_alert(
                    alert_type="invalid_base64_signature",
                    details={"round": server_round, "client_id": client_id}
                )
                rejected_count += 1
                continue

            # Check 5: Ed25519 signature verification
            manifest = {
                "round_id": int(round_id),
                "client_id": client_id,
                "num_samples": int(fit_res.num_examples),
                "model_hash": model_hash,
                "protocol_version": protocol_version,
            }

            if self.signature_validator.validate_update(client_id, signature, manifest):
                # FIX: Log accepted updates to audit trail
                self.audit_log.log_client_update(
                    round_number=server_round,
                    client_id=client_id,
                    num_samples=int(fit_res.num_examples),
                    signature_valid=True,
                )
                filtered.append((client_proxy, fit_res))
            else:
                self.audit_log.log_security_alert(
                    alert_type="invalid_signature",
                    details={"round": server_round, "client_id": client_id}
                )
                rejected_count += 1

        # Log round validation summary
        self.audit_log.log_event(
            event_type="round_validation_summary",
            details={
                "round": server_round,
                "accepted": len(filtered),
                "rejected": rejected_count,
                "total": len(results),
            },
            severity="INFO" if rejected_count == 0 else "WARNING"
        )

        return filtered

    def aggregate_fit(self, server_round, results, failures):
        results = self._validate_and_filter_results(server_round, results)

        if len(results) == 0:
            print(f"[Signing] No valid signed updates at round {server_round}; skipping aggregation")
            self.audit_log.log_event(
                event_type="aggregation_skipped",
                details={"round": server_round, "reason": "no_valid_updates"},
                severity="ERROR"
            )
            return None, {}

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Log successful aggregation
        self.audit_log.log_event(
            event_type="aggregation_complete",
            details={
                "round": server_round,
                "num_clients": len(results),
            },
            severity="INFO"
        )

        if aggregated_parameters is not None:
            print(f"\n---> [Round {server_round}] Extracting weights to save global model...")

            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            model = get_model()
            set_parameters(model, ndarrays)

            save_path = config.MODEL_WITH_DP if self.use_dp else config.MODEL_NO_DP
            save_model(model, save_path)

            print(f"---> SUCCESS: Final global model saved to {save_path}!\n")

            # FIX: Log model checkpoint to audit trail
            import hashlib
            model_hash = hashlib.sha256(
                b"".join(arr.tobytes() for arr in ndarrays)
            ).hexdigest()
            self.audit_log.log_model_checkpoint(
                round_number=server_round,
                model_path=save_path,
                model_hash=model_hash,
            )

            # FIX: Verify audit log integrity at end of training
            print("[Server] Verifying server audit log integrity...")
            self.audit_log.verify_integrity()

        return aggregated_parameters, aggregated_metrics


def weighted_average(metrics: list) -> dict:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def start_server(use_dp):
    print(f"Starting the Server... (Version B / DP enabled: {use_dp})")

    from src.privacy.secure_agg import get_secagg_strategy, has_legacy_secagg_strategy, has_workflow_secagg

    # FIX: Clearly warn about SecAgg status so the team knows what is and isn't active
    if has_legacy_secagg_strategy():
        print("[SecAgg] Legacy SecAggPlusStrategy available — wrapping strategy.")
    elif has_workflow_secagg():
        print("[SecAgg] WARNING: Current Flower version uses workflow/mod API for SecAgg.")
        print("[SecAgg] WARNING: Strategy-based SecAgg is NOT active in this run.")
        print("[SecAgg] WARNING: True SecAgg requires ServerApp/ClientApp migration (known next step).")
        print("[SecAgg] Continuing with signature validation + DP as active privacy layers.")
    else:
        print("[SecAgg] WARNING: No SecAgg API detected. Running without secure aggregation.")

    strategy = get_secagg_strategy(
        base_strategy_class=SaveModelStrategy,
        use_dp=use_dp,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.MIN_CLIENTS,
        min_evaluate_clients=config.MIN_CLIENTS,
        min_available_clients=config.MIN_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    history = fl.server.start_server(
        server_address=config.SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy,
    )

    metrics_file = "fl_with_dp_results.json" if use_dp else "fl_no_dp_results.json"
    os.makedirs(config.METRICS_DIR, exist_ok=True)

    with open(os.path.join(config.METRICS_DIR, metrics_file), "w") as f:
        json.dump({"rounds": config.NUM_ROUNDS, "dp": use_dp, "success": True}, f)

    print(f"Server is done! Metrics saved to {metrics_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server")
    parser.add_argument("--use_dp", action="store_true")
    args = parser.parse_args()

    start_server(args.use_dp)
