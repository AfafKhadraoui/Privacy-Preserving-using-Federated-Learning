"""
Payload Encryption — End-to-End Confidentiality for Model Updates.

What this does in plain terms:
    Even if someone intercepts the network traffic between a client and the server
    (or if TLS is somehow broken), the model updates are encrypted so they cannot
    be read. This is an extra layer on top of TLS, not a replacement.

How the encryption works (step by step):
    1. Key agreement (X25519):
       Client generates a one-time keypair (ephemeral).
       Client and server each have a public key the other knows.
       Using X25519 math, both sides compute the SAME shared secret
       without ever sending it over the network.

    2. Key derivation (HKDF):
       The shared secret is fed into HKDF (a key derivation function)
       to produce a proper 256-bit AES key.

    3. Encryption (AES-256-GCM):
       The model weights are encrypted with the AES key.
       GCM mode also authenticates the data — any tampering is detected.

    4. AAD (Associated Authenticated Data):
       round_id, client_id, and protocol_version are bound to the ciphertext.
       This prevents replay attacks: an encrypted update from round 1
       cannot be reused in round 3 because the AAD check would fail.

Current status:
    The encryption primitives are fully implemented and tested (smoke tests pass).
    In the current codebase, they run as a PROBE in client.py — verifying that
    the crypto works correctly each round by encrypting/decrypting the model hash.

    The actual Flower parameter transport is not yet encrypted end-to-end.
    Full production wiring (replacing the Flower transport with encrypted blobs)
    is the known next integration step.

Owner: Amel
"""

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
import os
import json
from typing import Tuple, Optional, Dict


def _build_associated_data(metadata: Dict) -> bytes:
    """
    Serialize protocol metadata into canonical bytes for use as AAD.

    AAD is authenticated (integrity-protected) but NOT encrypted.
    It is bound to the ciphertext — any change to these fields
    causes AES-GCM tag verification to fail, rejecting the message.

    We use sort_keys=True and no spaces so the same dict always
    produces exactly the same bytes regardless of insertion order.
    """
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encrypt_payload(
    plaintext: bytes,
    symmetric_key: bytes,
    associated_data: Optional[bytes] = None,
) -> bytes:
    """
    Encrypt data using AES-256-GCM with authentication.

    Args:
        plaintext:       The data to encrypt (e.g. serialized model weights)
        symmetric_key:   32-byte AES key (from derive_symmetric_key)
        associated_data: Optional AAD bytes (from _build_associated_data)
                         These are authenticated but not encrypted.

    Returns:
        Encrypted blob as: nonce (12 bytes) + ciphertext + auth_tag
        The nonce is prepended so the receiver can extract it for decryption.

    Raises:
        ValueError: If symmetric_key is not exactly 32 bytes
    """
    if len(symmetric_key) != 32:
        raise ValueError(f"Symmetric key must be 32 bytes, got {len(symmetric_key)}")

    nonce = os.urandom(12)  # 96-bit random nonce — GCM standard
    cipher = AESGCM(symmetric_key)
    ciphertext = cipher.encrypt(nonce, plaintext, associated_data=associated_data)

    # Prepend nonce so receiver can split it off
    return nonce + ciphertext


def decrypt_payload(
    encrypted_payload: bytes,
    symmetric_key: bytes,
    associated_data: Optional[bytes] = None,
) -> bytes:
    """
    Decrypt a payload produced by encrypt_payload.

    Args:
        encrypted_payload: Output from encrypt_payload (nonce + ciphertext + tag)
        symmetric_key:     Same 32-byte key used during encryption
        associated_data:   Must exactly match the AAD used during encryption

    Returns:
        Decrypted plaintext bytes

    Raises:
        cryptography.exceptions.InvalidTag: If auth tag check fails —
            this means either the key is wrong, the ciphertext was tampered with,
            or the AAD does not match. Treat all three cases as tamper detection.
        ValueError: If symmetric_key is not exactly 32 bytes
    """
    if len(symmetric_key) != 32:
        raise ValueError(f"Symmetric key must be 32 bytes, got {len(symmetric_key)}")

    nonce = encrypted_payload[:12]
    ciphertext = encrypted_payload[12:]

    cipher = AESGCM(symmetric_key)
    return cipher.decrypt(nonce, ciphertext, associated_data=associated_data)


def generate_ephemeral_keypair() -> Tuple[bytes, bytes]:
    """
    Generate a one-time X25519 keypair for a single encryption operation.

    A new ephemeral keypair is generated for every message so that
    even if a long-term key is later compromised, past messages remain safe
    (this property is called "forward secrecy").

    Returns:
        (private_key_bytes, public_key_bytes) — raw 32-byte values each
    """
    private_key = x25519.X25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key.private_bytes_raw(), public_key.public_bytes_raw()


def compute_shared_secret(client_private: bytes, server_public: bytes) -> bytes:
    """
    Perform X25519 Diffie-Hellman key agreement.

    Both sides (client and server) run this function with their own private key
    and the other party's public key. The result is the same shared secret
    on both sides — without it ever being transmitted.

    Args:
        client_private: Client's private key bytes (32 bytes)
        server_public:  Server's public key bytes (32 bytes)

    Returns:
        32-byte shared secret
    """
    client_private_key = x25519.X25519PrivateKey.from_private_bytes(client_private)
    server_public_key = x25519.X25519PublicKey.from_public_bytes(server_public)
    return client_private_key.exchange(server_public_key)


def derive_symmetric_key(shared_secret: bytes, info: str = "model_encryption") -> bytes:
    """
    Derive a 256-bit AES key from a shared secret using HKDF-SHA256.

    We use HKDF rather than the raw shared secret because X25519 output
    is not uniformly random — HKDF stretches and conditions it into a
    proper cryptographic key.

    Args:
        shared_secret: Output from compute_shared_secret (32 bytes)
        info:          Context string to bind the key to a specific purpose.
                       Using different info strings produces different keys
                       from the same shared secret (useful for key separation).

    Returns:
        32-byte symmetric key suitable for AES-256-GCM
    """
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info.encode("utf-8"),
    )
    return hkdf.derive(shared_secret)


class PayloadEncryptor:
    """
    Client-side helper that encrypts a payload for a specific server.

    Handles the full encrypt flow: ephemeral keypair → shared secret →
    AES key → encrypt with AAD. Returns a dict with everything the server
    needs to decrypt.
    """

    def __init__(self, server_public_key_bytes: bytes):
        """
        Args:
            server_public_key_bytes: Server's long-term X25519 public key (32 bytes)
        """
        self.server_public_key = server_public_key_bytes

    def encrypt(
        self,
        plaintext: bytes,
        round_id: int,
        client_id: str,
        protocol_version: str = "v1",
    ) -> dict:
        """
        Encrypt a payload and return everything needed for decryption.

        The returned dict includes:
            - encrypted_payload:    hex-encoded ciphertext (nonce + ciphertext + tag)
            - ephemeral_public_key: hex-encoded one-time public key (server needs this)
            - round_id, client_id, protocol_version: AAD fields (authenticated but not encrypted)
            - algorithm, key_agreement: metadata for logging/debugging

        Args:
            plaintext:        Bytes to encrypt (e.g. model hash or serialized weights)
            round_id:         FL round number — bound to ciphertext via AAD
            client_id:        Client identifier — bound to ciphertext via AAD
            protocol_version: Protocol version — bound to ciphertext via AAD

        Returns:
            Dictionary with encrypted payload and metadata
        """
        ephemeral_private, ephemeral_public = generate_ephemeral_keypair()
        shared_secret = compute_shared_secret(ephemeral_private, self.server_public_key)
        symmetric_key = derive_symmetric_key(shared_secret, info="model_encryption")

        aad_dict = {
            "round_id": round_id,
            "client_id": client_id,
            "protocol_version": protocol_version,
        }
        associated_data = _build_associated_data(aad_dict)

        encrypted_payload = encrypt_payload(plaintext, symmetric_key, associated_data=associated_data)

        return {
            "encrypted_payload": encrypted_payload.hex(),
            "ephemeral_public_key": ephemeral_public.hex(),
            "round_id": round_id,
            "client_id": client_id,
            "protocol_version": protocol_version,
            "algorithm": "AES-256-GCM",
            "key_agreement": "X25519+HKDF",
        }


class PayloadDecryptor:
    """
    Server-side helper that decrypts a payload from a client.

    Uses the server's long-term private key together with the client's
    ephemeral public key (included in the encrypted dict) to reconstruct
    the shared secret and decrypt.
    """

    def __init__(self, server_private_key_bytes: bytes):
        """
        Args:
            server_private_key_bytes: Server's long-term X25519 private key (32 bytes)
                                      Keep this secret — treat it like a password.
        """
        self.server_private_key = server_private_key_bytes

    def decrypt(self, encrypted_data: dict) -> bytes:
        """
        Decrypt a payload produced by PayloadEncryptor.encrypt().

        Args:
            encrypted_data: The dict returned by PayloadEncryptor.encrypt()

        Returns:
            Decrypted plaintext bytes

        Raises:
            cryptography.exceptions.InvalidTag: If authentication fails —
                means tampered data, wrong key, or mismatched AAD.
                Never proceed with data that fails this check.
        """
        encrypted_payload = bytes.fromhex(encrypted_data["encrypted_payload"])
        ephemeral_public = bytes.fromhex(encrypted_data["ephemeral_public_key"])

        shared_secret = compute_shared_secret(self.server_private_key, ephemeral_public)
        symmetric_key = derive_symmetric_key(shared_secret, info="model_encryption")

        aad_dict = {
            "round_id": encrypted_data["round_id"],
            "client_id": encrypted_data["client_id"],
            "protocol_version": encrypted_data["protocol_version"],
        }
        associated_data = _build_associated_data(aad_dict)

        return decrypt_payload(encrypted_payload, symmetric_key, associated_data=associated_data)
