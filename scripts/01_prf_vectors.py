#!/usr/bin/env python3
"""Freeze the PRF test vectors. Regenerating them is a breaking change: the vectors
are the cross-platform determinism contract, so a diff here means every previously
published number used a different g-value sequence.

    python scripts/01_prf_vectors.py --seed 0 [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wmk.config import canonical_json
from wmk.keys import MASTER_SEED_HEX, null_key
from wmk.prf import g_bit

OUT = Path(__file__).resolve().parents[1] / "tests" / "vectors" / "prf.json"
N_VECTORS = 1000
UINT32_MAX = 0xFFFFFFFF
ROUND_MAX = 0xFFFF


def build(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vectors = []

    edges = [
        (0, (), 0, 0),
        (0, (0,), 0, 0),
        (1, (UINT32_MAX,), UINT32_MAX, ROUND_MAX),
        (2, (0, UINT32_MAX, 0, UINT32_MAX), UINT32_MAX, 1),
        (3, tuple(range(8)), 7, 8),
    ]
    for key_idx, context, candidate, round_idx in edges:
        key = null_key(key_idx)
        vectors.append(
            {
                "key_index": key_idx,
                "context": list(context),
                "candidate": candidate,
                "round_idx": round_idx,
                "g": g_bit(key, context, candidate, round_idx),
            }
        )

    while len(vectors) < N_VECTORS:
        key_idx = int(rng.integers(0, 10))
        depth = int(rng.integers(1, 9))
        context = [int(v) for v in rng.integers(0, 152064, size=depth)]
        candidate = int(rng.integers(0, 152064))
        round_idx = int(rng.integers(0, 16))
        vectors.append(
            {
                "key_index": key_idx,
                "context": context,
                "candidate": candidate,
                "round_idx": round_idx,
                "g": g_bit(null_key(key_idx), context, candidate, round_idx),
            }
        )

    return {
        "schema_version": 1,
        "seed": seed,
        "prf": "hmac-sha256",
        "encoding": "round_idx:u16be || candidate:u32be || context tokens:u32be oldest first",
        "g_extraction": "least significant bit of byte 0",
        "master_seed_hex": MASTER_SEED_HEX,
        "n_vectors": len(vectors),
        "vectors": vectors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="fail if the file would change")
    args = ap.parse_args()

    payload = canonical_json(build(args.seed)) + "\n"

    if args.check:
        if not OUT.exists():
            print(f"MISSING {OUT}")
            return 1
        if OUT.read_text(encoding="ascii") != payload:
            print(f"CHANGED {OUT} -- frozen vectors would differ; this breaks C7")
            return 1
        print(f"OK {OUT} ({json.loads(payload)['n_vectors']} vectors)")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(payload, encoding="ascii")
    ones = sum(v["g"] for v in json.loads(payload)["vectors"])
    print(f"wrote {OUT} ({N_VECTORS} vectors, {ones} ones)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
