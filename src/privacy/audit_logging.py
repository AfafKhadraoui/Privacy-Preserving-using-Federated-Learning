"""
Audit Logging — Tamper-Evident Privacy Event Logging.

This module maintains a cryptographically linked audit log of all privacy-relevant
events (training rounds, DP activation, epsilon budget, model checkpoints, etc.).

The log is tamper-evident: any modification of a previous entry will be detectable
because the hash chain breaks.

How it works:
    Every entry contains a SHA-256 hash of itself plus the hash of the previous entry.
    This creates a chain — if you change entry 3, its hash changes, which breaks
    entry 4's "previous_hash" check, and so on. You cannot silently edit past entries.

Owner: Amel
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any
import os


class AuditLog:
    """Tamper-evident cryptographic audit log."""

    def __init__(self, log_path: str):
        """
        Args:
            log_path: File path where the log will be written (e.g. results/metrics/audit.log)

        Each call to __init__ starts a FRESH chain for this run.
        Old log files are rotated (renamed with a timestamp) so past runs are
        preserved but do not interfere with the new run's hash chain.
        """
        self.log_path = log_path
        self.entries = []
        self.previous_hash = "0" * 64  # Genesis hash — starting point of every fresh chain

        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        self._rotate_old_log()

    def _rotate_old_log(self) -> None:
        """
        Rotate any existing log file from a previous run.

        WHY THIS MATTERS — The chain-breaking bug:
            The old code appended new entries to the existing file but reset
            previous_hash to the genesis hash (000...0). So the NEW run's first
            entry would have previous_hash=000...0, but the last OLD entry had
            a real hash. When verify_integrity walked from entry 0 it expected
            each entry's previous_hash to equal the hash of the entry before it.
            The first new entry would fail this check → "Chain broken at entry N".

        THE FIX:
            Rotate (rename) the old log so each run starts with an empty file
            and a clean chain starting from genesis. Old logs are kept for
            auditing purposes — they just live in a separate file.
        """
        if not os.path.exists(self.log_path):
            return

        # Rename the old log to <name>.<timestamp>.bak so it is preserved
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.log_path}.{timestamp}.bak"
        try:
            os.rename(self.log_path, backup_path)
            print(f"[Audit] Rotated previous log → {os.path.basename(backup_path)}")
        except OSError as exc:
            # If rotation fails (e.g. file locked), truncate in place as fallback
            print(f"[Audit] WARNING: Could not rotate old log ({exc}). Truncating in place.")
            open(self.log_path, "w").close()

    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "INFO"
    ) -> str:
        """
        Log an event with cryptographic linking.

        Args:
            event_type: Type of event e.g. "client_update", "dp_enabled", "model_checkpoint"
            details: Event-specific data dictionary
            severity: One of "DEBUG", "INFO", "WARNING", "ERROR"

        Returns:
            SHA-256 hash of this log entry (can be used to verify this entry later)
        """
        entry = {
            "sequence": len(self.entries),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "severity": severity,
            "details": details,
            "previous_hash": self.previous_hash,
        }

        # Hash does NOT include the "hash" field itself (computed before adding it)
        entry_json = json.dumps(entry, sort_keys=True, separators=(',', ':'))
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()
        entry["hash"] = entry_hash

        # Persist the full entry including hash to disk
        persisted_entry = json.dumps(entry, sort_keys=True, separators=(',', ':'))

        self.entries.append(entry)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(persisted_entry + "\n")

        self.previous_hash = entry_hash
        self._print_entry(entry)

        return entry_hash

    def _print_entry(self, entry: Dict):
        """Print a log entry to console in a readable format."""
        ts = entry["timestamp"].split("T")[1][:8]  # HH:MM:SS only
        severity = entry["severity"]
        event = entry["event_type"]
        print(f"[Audit] [{ts}] [{severity}] [{event}] {entry['details']}")

    def verify_integrity(self) -> bool:
        """
        Verify the full log has not been tampered with by re-checking the hash chain.

        Reads directly from the file on disk (not just in-memory) so it catches
        tampering that happened while the program was not running.

        Returns:
            True if everything checks out, False if any tampering is detected
        """
        print(f"[Audit] Verifying integrity of: {self.log_path}")

        previous = "0" * 64
        file_entries = []

        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        file_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"[Audit] ERROR: Invalid JSON at line {line_number} — log may be corrupted")
                        return False

        entries_to_check = file_entries if file_entries else self.entries

        if len(entries_to_check) == 0:
            print("[Audit] WARNING: Log is empty — nothing to verify")
            return True

        for i, entry in enumerate(entries_to_check):
            # Check hash field exists
            if "hash" not in entry:
                print(f"[Audit] ERROR: Missing hash field at entry {i} — log is corrupted")
                return False

            # Check chain link
            if entry["previous_hash"] != previous:
                print(f"[Audit] ERROR: Chain broken at entry {i} — tampering detected")
                return False

            # Recompute and compare hash
            entry_copy = entry.copy()
            stored_hash = entry_copy.pop("hash")
            entry_json = json.dumps(entry_copy, sort_keys=True, separators=(',', ':'))
            computed_hash = hashlib.sha256(entry_json.encode()).hexdigest()

            if computed_hash != stored_hash:
                print(f"[Audit] ERROR: Hash mismatch at entry {i} (event: {entry.get('event_type', '?')}) — content was modified")
                return False

            previous = stored_hash

        print(f"[Audit] OK: Integrity verified — {len(entries_to_check)} entries, chain intact")
        return True


class PrivacyAuditLog(AuditLog):
    """
    Specialized audit log with convenience methods for FL privacy events.

    Use this instead of AuditLog directly — it has named methods for every
    important event type so you don't have to remember the event_type strings.
    """

    def log_fl_round(self, round_number: int, num_clients: int, num_samples: int):
        """Log the start of an FL training round."""
        self.log_event(
            event_type="fl_round_start",
            details={
                "round": round_number,
                "num_clients": num_clients,
                "num_samples": num_samples,
            },
            severity="INFO"
        )

    def log_dp_enabled(self, noise_multiplier: float, max_grad_norm: float):
        """Log that DP-SGD is active for this training run."""
        self.log_event(
            event_type="dp_enabled",
            details={
                "noise_multiplier": noise_multiplier,
                "max_grad_norm": max_grad_norm,
                "mechanism": "DP-SGD (Opacus)",
            },
            severity="INFO"
        )

    def log_epsilon_update(self, round_number: int, cumulative_epsilon: float, delta: float):
        """Log how much privacy budget has been spent after a round."""
        self.log_event(
            event_type="epsilon_update",
            details={
                "round": round_number,
                "cumulative_epsilon": cumulative_epsilon,
                "delta": delta,
            },
            severity="INFO"
        )

    def log_model_checkpoint(self, round_number: int, model_path: str, model_hash: str):
        """Log when a model is saved to disk with its SHA-256 hash for integrity."""
        self.log_event(
            event_type="model_checkpoint",
            details={
                "round": round_number,
                "path": model_path,
                "sha256": model_hash,
            },
            severity="INFO"
        )

    def log_client_update(self, round_number: int, client_id: str, num_samples: int, signature_valid: bool):
        """Log receipt of a client update and whether its signature was valid."""
        self.log_event(
            event_type="client_update_received",
            details={
                "round": round_number,
                "client_id": client_id,
                "num_samples": num_samples,
                "signature_valid": signature_valid,
            },
            severity="INFO" if signature_valid else "WARNING"
        )

    def log_security_alert(self, alert_type: str, details: Dict):
        """Log a security event — rejected update, signature failure, protocol mismatch, etc."""
        self.log_event(
            event_type="security_alert",
            details={"alert_type": alert_type, **details},
            severity="WARNING"
        )

    def log_attack_simulation(self, attack_type: str, target_model: str, result: str):
        """Log when an attack is simulated for evaluation purposes."""
        self.log_event(
            event_type="attack_simulation",
            details={
                "attack_type": attack_type,
                "target_model": target_model,
                "result": result,
            },
            severity="INFO"
        )


# ---------------------------------------------------------------------------
# Global instance management
# FIX: Changed from a single global to a per-path registry so that the server
# and each client can have their own separate audit log without overwriting
# each other's instance when running in the same process (e.g. local simulation).
# ---------------------------------------------------------------------------
_audit_log_registry: Dict[str, PrivacyAuditLog] = {}


def get_audit_log(log_path: str = "results/metrics/audit.log") -> PrivacyAuditLog:
    """
    Get or create an audit log for the given path.

    Each unique path gets its own instance, so server and clients
    don't interfere with each other in local simulations.
    """
    global _audit_log_registry
    if log_path not in _audit_log_registry:
        _audit_log_registry[log_path] = PrivacyAuditLog(log_path)
    return _audit_log_registry[log_path]
