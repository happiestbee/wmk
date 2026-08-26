"""One frozen-vector test subsumes determinism, bit range, encoding order and
platform stability. The PRF fast path must agree with the reference bit-for-bit or
the empirical null and the detector are measuring different functions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wmk.keys import MASTER_SEED_HEX, MASTER_SEED_PREIMAGE, N_NULL_KEYS, null_key, null_keys
from wmk.prf import PRF, encode_message, g_bit, g_bits

VECTORS = json.loads(
    (Path(__file__).resolve().parent / "vectors" / "prf.json").read_text(encoding="ascii")
)


def test_frozen_vectors_reproduce_exactly():
    mismatches = []
    for i, v in enumerate(VECTORS["vectors"]):
        key = null_key(v["key_index"])
        got = g_bit(key, v["context"], v["candidate"], v["round_idx"])
        if got != v["g"]:
            mismatches.append((i, v, got))
    assert not mismatches, f"{len(mismatches)} PRF vectors diverged, first: {mismatches[0]}"


def test_fast_path_matches_reference():
    for v in VECTORS["vectors"][:200]:
        prf = PRF(null_key(v["key_index"]))
        assert prf.bit(v["context"], v["candidate"], v["round_idx"]) == v["g"]

    key = null_key(0)
    context, candidate = (11, 22, 33, 44), 5555
    assert np.array_equal(
        PRF(key).bits(context, candidate, 8), g_bits(key, context, candidate, 8)
    )


def test_encoding_is_the_documented_byte_layout():
    assert encode_message((0x01020304,), 0x0A0B0C0D, 0x1234) == bytes.fromhex(
        "1234" "0A0B0C0D" "01020304"
    )
    with pytest.raises(ValueError):
        encode_message((), 0, 0x10000)
    with pytest.raises(ValueError):
        encode_message((-1,), 0, 0)
    with pytest.raises(TypeError):
        encode_message((), 1.0, 0)


def test_null_keys_derive_from_the_published_seed():
    assert hashlib.sha256(MASTER_SEED_PREIMAGE).hexdigest() == MASTER_SEED_HEX
    assert VECTORS["master_seed_hex"] == MASTER_SEED_HEX

    keys = null_keys()
    assert len(keys) == N_NULL_KEYS
    assert len({bytes(k) for k in keys}) == N_NULL_KEYS
    assert all(len(k) == 32 for k in keys)
    assert null_key(0) == keys[0]
    with pytest.raises(ValueError):
        null_key(N_NULL_KEYS)
