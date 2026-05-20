"""Shared CV utilities and encrypted storage."""

import base64
import logging
import os
from typing import Any, Dict

import numpy as np

from src.config import ENCRYPTION_KEY_ENV

logger = logging.getLogger(__name__)


# -- Encryption helpers ---------------------------------------------------

def _get_encryption_key() -> bytes:
    """Derive a 32-byte Fernet-compatible key from the environment variable."""
    try:
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
    except ImportError:
        raise ImportError(
            "cryptography package is required for encrypted storage. "
            "Install it with: pip install cryptography"
        )

    key_material = os.getenv(ENCRYPTION_KEY_ENV, "")
    if not key_material:
        logger.warning(
            "No encryption key found in %s. Using default key. "
            "Set %s for production use.",
            ENCRYPTION_KEY_ENV, ENCRYPTION_KEY_ENV,
        )
        key_material = "eventguard-default-dev-key"

    salt = b"eventguard-salt-v1"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    derived = kdf.derive(key_material.encode("utf-8"))
    return base64.urlsafe_b64encode(derived)


def encrypt_data(data: bytes) -> bytes:
    """Encrypt data using Fernet (AES-128-CBC under the hood)."""
    from cryptography.fernet import Fernet

    key = _get_encryption_key()
    f = Fernet(key)
    return f.encrypt(data)


def decrypt_data(token: bytes) -> bytes:
    """Decrypt Fernet-encrypted data."""
    from cryptography.fernet import Fernet

    key = _get_encryption_key()
    f = Fernet(key)
    return f.decrypt(token)


# -- Legacy load/save kept for migration compatibility --------------------

def load_db(path) -> Dict[str, Any]:
    """Load legacy pickle database (kept for backward compat)."""
    logger.warning("Using legacy pickle load from %s", path)
    import pickle

    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run ingest.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def save_db(data: Dict[str, Any], path):
    """Save legacy pickle database (kept for backward compat)."""
    logger.warning("Using legacy pickle save to %s", path)
    import pickle

    with open(path, "wb") as f:
        pickle.dump(data, f)
