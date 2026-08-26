"""Keyed PRF for tournament sampling.

Canonical encoding, frozen by tests/vectors/prf.json:
    round_idx uint16 BE || candidate uint32 BE || context tokens uint32 BE, oldest first
g is the least significant bit of byte 0 of HMAC-SHA256(key, encoding).
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Sequence

import numpy as np

__all__ = ["encode_message", "g_bit", "g_bits", "PRF", "context_at", "MAX_ROUNDS"]

MAX_ROUNDS = 64

_ROUND_MAX = 0xFFFF
_TOKEN_MAX = 0xFFFFFFFF
_SHA256_BLOCK = 64


def _check_token(value: int, what: str, limit: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{what} must be an int, got {type(value).__name__}")
    value = int(value)
    if not 0 <= value <= limit:
        raise ValueError(f"{what} {value} out of range 0..{limit}")
    return value


def encode_message(context: Sequence[int], candidate: int, round_idx: int) -> bytes:
    out = bytearray()
    out += _check_token(round_idx, "round_idx", _ROUND_MAX).to_bytes(2, "big")
    out += _check_token(candidate, "candidate", _TOKEN_MAX).to_bytes(4, "big")
    for tok in context:
        out += _check_token(tok, "context token", _TOKEN_MAX).to_bytes(4, "big")
    return bytes(out)


def g_bit(key: bytes, context: Sequence[int], candidate: int, round_idx: int) -> int:
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError(f"key must be bytes, got {type(key).__name__}")
    msg = encode_message(context, candidate, round_idx)
    return hmac.new(bytes(key), msg, hashlib.sha256).digest()[0] & 1


def g_bits(key: bytes, context: Sequence[int], candidate: int, n_rounds: int) -> np.ndarray:
    if not 1 <= n_rounds <= MAX_ROUNDS:
        raise ValueError(f"n_rounds must be in 1..{MAX_ROUNDS}, got {n_rounds}")
    return np.fromiter(
        (g_bit(key, context, candidate, r) for r in range(n_rounds)),
        dtype=np.uint8,
        count=n_rounds,
    )


class PRF:
    """g-value oracle with the key's HMAC pad blocks pre-absorbed.

    Byte-identical to g_bit by the RFC 2104 decomposition; test_prf_vectors.py
    asserts that over the frozen vectors.
    """

    __slots__ = ("_inner", "_outer", "key")

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError(f"key must be bytes, got {type(key).__name__}")
        self.key = bytes(key)
        block = hashlib.sha256(self.key).digest() if len(self.key) > _SHA256_BLOCK else self.key
        block = block.ljust(_SHA256_BLOCK, b"\x00")
        self._inner = hashlib.sha256(bytes(b ^ 0x36 for b in block))
        self._outer = hashlib.sha256(bytes(b ^ 0x5C for b in block))

    def bit(self, context: Sequence[int], candidate: int, round_idx: int) -> int:
        return self._digest0(encode_message(context, candidate, round_idx)) & 1

    def bits(self, context: Sequence[int], candidate: int, n_rounds: int) -> np.ndarray:
        """Rounds 0..n_rounds-1, so one pass serves the whole d grid: a d=1 scorer
        reads bit 0, a d=8 scorer reads bits 0..7."""
        if not 1 <= n_rounds <= MAX_ROUNDS:
            raise ValueError(f"n_rounds must be in 1..{MAX_ROUNDS}, got {n_rounds}")
        tail = encode_message(context, candidate, 0)[2:]
        out = np.empty(n_rounds, dtype=np.uint8)
        for r in range(n_rounds):
            out[r] = self._digest0(r.to_bytes(2, "big") + tail) & 1
        return out

    def _digest0(self, msg: bytes) -> int:
        inner = self._inner.copy()
        inner.update(msg)
        outer = self._outer.copy()
        outer.update(inner.digest())
        return outer.digest()[0]

    def __repr__(self) -> str:
        return f"PRF(sha256(key)={hashlib.sha256(self.key).hexdigest()[:8]})"


def context_at(token_ids: Sequence[int], position: int, context_depth: int) -> tuple[int, ...]:
    """Empty tuple when the position lacks a full context window; such positions are
    unscorable and are excluded by wmk.masking.scorable_mask."""
    if position < context_depth:
        return ()
    return tuple(int(t) for t in token_ids[position - context_depth: position])
