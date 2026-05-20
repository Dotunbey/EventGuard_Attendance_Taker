"""Unit tests for src/utils.py functions."""

import numpy as np
import pytest

from src.utils import (
    encrypt_data,
    decrypt_data,
)


class TestEncryption:
    def test_roundtrip(self):
        data = b"hello biometric data"
        encrypted = encrypt_data(data)
        assert encrypted != data
        decrypted = decrypt_data(encrypted)
        assert decrypted == data

    def test_different_data_different_ciphertext(self):
        a = encrypt_data(b"data_a")
        b = encrypt_data(b"data_b")
        assert a != b
