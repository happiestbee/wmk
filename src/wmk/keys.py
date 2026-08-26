"""HKDF-SHA256 derivation of the null-key ensemble.

The master seed is nothing-up-my-sleeve: sha256(b"wmk-watermark-capacity-atlas-v1").
test_keys.py re-derives it, so a substituted seed fails rather than silently changing
the published null.
"""

from __future__ import annotations

import hashlib
import hmac

__all__ = [
    "MASTER_SEED_HEX",
    "MASTER_SEED_PREIMAGE",
    "N_NULL_KEYS",
    "hkdf",
    "null_key",
    "null_keys",
]

MASTER_SEED_PREIMAGE = b"wmk-watermark-capacity-atlas-v1"
MASTER_SEED_HEX = "daa9c0f6d14873012705bd9f30f9f45e7ea2537cabcf4f6664bc40786c9b26d3"
N_NULL_KEYS = 100

_HASH_LEN = 32


def hkdf(ikm: bytes, info: bytes, length: int = 32, salt: bytes = b"") -> bytes:
    """RFC 5869 extract-then-expand."""
    if length < 1 or length > 255 * _HASH_LEN:
        raise ValueError(f"length {length} out of range")
    prk = hmac.new(salt or bytes(_HASH_LEN), ikm, hashlib.sha256).digest()
    okm = b""
    block = b""
    counter = 1
    while len(okm) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
        counter += 1
    return okm[:length]


def master_seed() -> bytes:
    seed = bytes.fromhex(MASTER_SEED_HEX)
    if seed != hashlib.sha256(MASTER_SEED_PREIMAGE).digest():
        raise ValueError("MASTER_SEED_HEX does not match its published preimage")
    return seed


def null_key(index: int) -> bytes:
    if not 0 <= index < N_NULL_KEYS:
        raise ValueError(f"null key index {index} out of range 0..{N_NULL_KEYS - 1}")
    return hkdf(master_seed(), info=b"wmk-null-key-%d" % index, length=32)


def null_keys(n: int = N_NULL_KEYS) -> list[bytes]:
    return [null_key(i) for i in range(n)]
