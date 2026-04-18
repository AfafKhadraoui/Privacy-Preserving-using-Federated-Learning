"""
Cryptographic Signing and Verification for Model Update Integrity.

What this does in plain terms:
    Every time a client sends a model update to the server, it attaches
    a digital signature proving:
        1. The update really came from that specific client (authenticity)
        2. The update has not been modified in transit (integrity)

    The server verifies every signature before accepting any update.
    If the signature is missing, wrong, or doesn't match — the update is rejected.

How Ed25519 signing works:
    - Each client generates a keypair: a private key (secret) and a public key (shareable)
    - The private key is used to sign the update — only that client can do this
    - The public key is used to verify the signature — anyone can do this
    - Mathematically impossible to fake a signature without the private key

Key storage:
    Keys are saved as PEM files in the KEYS_DIR folder defined in config.py.
    IMPORTANT: Add KEYS_DIR to .gitignore — never commit private keys to GitHub.
    IMPORTANT: Set the SIGNING_KEY_PASSPHRASE environment variable before running.
               Do NOT rely on the default fallback passphrase in production.

Owner: Amel
"""

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature
import os
import json


def _get_signing_passphrase() -> bytes:
    """
    Get the passphrase used to encrypt private key PEM files.

    Reads from the SIGNING_KEY_PASSPHRASE environment variable.

    IMPORTANT: The fallback "secure_passphrase" is only for local development
    and demos. In any real deployment, always set SIGNING_KEY_PASSPHRASE in
    your environment or .env file. Never commit the passphrase to GitHub.

    How to set it:
        Windows:  $env:SIGNING_KEY_PASSPHRASE = "your-strong-passphrase"
        Linux/Mac: export SIGNING_KEY_PASSPHRASE="your-strong-passphrase"
        .env file: SIGNING_KEY_PASSPHRASE=your-strong-passphrase
    """
    env_value = os.getenv("SIGNING_KEY_PASSPHRASE")

    if env_value is None:
        print("[Signing] WARNING: SIGNING_KEY_PASSPHRASE env var not set — using insecure default")
        print("[Signing] WARNING: Set this env var before running in production")
        return b"secure_passphrase"

    return env_value.encode("utf-8")


def generate_client_keypair(client_id: str, output_dir: str = "data/keys"):
    """
    Generate an Ed25519 keypair for a client and save to PEM files.

    This only needs to be called once per client. After that, load_private_key()
    is used to reload the key each time.

    Args:
        client_id: Unique client identifier e.g. "client_00"
        output_dir: Directory to save the PEM files (should be in .gitignore)

    Returns:
        (private_key, public_key) — Ed25519 key objects
    """
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    os.makedirs(output_dir, exist_ok=True)

    # Save private key — encrypted with passphrase so it is safe at rest
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.BestAvailableEncryption(_get_signing_passphrase()),
    )
    private_path = os.path.join(output_dir, f"{client_id}_private.pem")
    with open(private_path, "wb") as f:
        f.write(private_pem)

    # Save public key — no encryption needed, safe to share with server
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_path = os.path.join(output_dir, f"{client_id}_public.pem")
    with open(public_path, "wb") as f:
        f.write(public_pem)

    print(f"[Signing] Generated keypair for {client_id}")
    print(f"  Private key: {private_path}")
    print(f"  Public key:  {public_path}")

    return private_key, public_key


def load_private_key(key_path: str, passphrase: bytes = None):
    """
    Load a private key from a PEM file.

    Args:
        key_path:   Path to the _private.pem file
        passphrase: Encryption passphrase. If None, reads from environment variable.

    Returns:
        Ed25519PrivateKey object
    """
    if passphrase is None:
        passphrase = _get_signing_passphrase()

    with open(key_path, "rb") as f:
        key_data = f.read()

    return serialization.load_pem_private_key(key_data, password=passphrase)


def load_public_key(key_path: str):
    """
    Load a public key from a PEM file.

    Args:
        key_path: Path to the _public.pem file

    Returns:
        Ed25519PublicKey object
    """
    with open(key_path, "rb") as f:
        key_data = f.read()

    return serialization.load_pem_public_key(key_data)


def sign_update(private_key, update_dict: dict) -> bytes:
    """
    Sign a model update dictionary using Ed25519.

    The update_dict is serialized to canonical JSON before signing,
    so the exact same dict always produces the same bytes to sign.

    Args:
        private_key: Ed25519PrivateKey (from generate_client_keypair or load_private_key)
        update_dict: Dictionary containing round_id, client_id, model_hash, etc.

    Returns:
        64-byte Ed25519 signature
    """
    message = json.dumps(update_dict, sort_keys=True).encode("utf-8")
    return private_key.sign(message)


def verify_update(public_key, signature: bytes, update_dict: dict) -> bool:
    """
    Verify that an update was signed with the correct private key.

    Returns False (does not raise) if the signature is invalid, so
    callers can handle it cleanly with a simple if/else.

    Args:
        public_key: Ed25519PublicKey for the claimed client
        signature:  64-byte signature bytes from sign_update()
        update_dict: The exact same dict that was passed to sign_update()

    Returns:
        True if signature is valid, False if invalid or tampered
    """
    message = json.dumps(update_dict, sort_keys=True).encode("utf-8")
    try:
        public_key.verify(signature, message)
        return True
    except InvalidSignature:
        return False


class SignatureValidator:
    """
    Server-side validator that verifies client update signatures.

    Usage:
        validator = SignatureValidator()
        validator.register_client_key("client_00", "data/keys/client_00_public.pem")
        is_valid = validator.validate_update("client_00", signature_bytes, manifest_dict)
    """

    def __init__(self):
        self.public_keys = {}  # {client_id: Ed25519PublicKey}

    def register_client_key(self, client_id: str, public_key_path: str):
        """
        Register a client's public key so we can verify their updates.

        Call this once per client at server startup, or lazily on first update.

        Args:
            client_id:       e.g. "client_00"
            public_key_path: Path to the client's _public.pem file
        """
        public_key = load_public_key(public_key_path)
        self.public_keys[client_id] = public_key
        print(f"[Signing] Registered public key for {client_id}")

    def validate_update(self, client_id: str, signature: bytes, update_dict: dict) -> bool:
        """
        Verify a client update is properly signed.

        Args:
            client_id:   The client claiming to have sent this update
            signature:   The signature bytes attached to the update
            update_dict: The manifest dict that was signed (must match exactly)

        Returns:
            True if valid and accepted, False if rejected
        """
        if client_id not in self.public_keys:
            print(f"[Signing] REJECTED {client_id}: no public key registered for this client")
            return False

        is_valid = verify_update(self.public_keys[client_id], signature, update_dict)

        if is_valid:
            print(f"[Signing] ACCEPTED {client_id}: signature valid")
        else:
            print(f"[Signing] REJECTED {client_id}: signature invalid — update may have been tampered with")

        return is_valid
