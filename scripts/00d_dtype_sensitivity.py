#!/usr/bin/env python3
"""Quantify what each C8-permitted dtype does to the measured quantity.

C8 forbids quantized loading because p_active is a functional of the next-token
distribution. It names fp16/bf16 as permitted, which presumes a GPU host. This script
measures the actual divergence in p_active and N_eff so the dtype choice on a CPU host
rests on numbers.

    python scripts/00d_dtype_sensitivity.py --model data/weights/Qwen__Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmk.capacity import p_active
from wmk.config import canonical_json
from wmk.registry import REPO_ROOT

PROBES = {
    "prose": (
        "The lamplighter came down the street at dusk, and one by one the windows "
        "answered him. It was the hour when the city seemed to hesitate between two "
        "lives, neither of which it had chosen for itself, and the river below carried "
        "the last of the light out toward the sea without comment or complaint."
    ),
    "factual": (
        "The Treaty of Westphalia was signed in 1648, ending the Thirty Years' War. "
        "It comprised two separate peace agreements concluded in the Westphalian cities "
        "of Osnabruck and Munster. The settlement established the principle that the "
        "sovereign of each state had authority over religious practice within its borders."
    ),
    "code": (
        "import numpy as np\n\n\n"
        "def normalize(rows: np.ndarray) -> np.ndarray:\n"
        "    totals = rows.sum(axis=-1, keepdims=True)\n"
        "    if not np.all(totals > 0):\n"
        "        raise ValueError('rows must have positive mass')\n"
        "    return rows / totals\n\n\n"
        "def entropy(rows: np.ndarray) -> np.ndarray:\n"
        "    p = normalize(rows)\n"
        "    return -np.sum(p * np.log(p, where=p > 0), axis=-1)\n"
    ),
}
DEPTHS = (1, 2, 4, 8)


def logits_for(model_dir: Path, dtype: torch.dtype, text: str, tok) -> tuple[np.ndarray, float]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    model.eval()
    ids = tok(text, return_tensors="pt").input_ids
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(ids).logits[0]
    elapsed = time.perf_counter() - t0
    del model
    return logits.float().numpy(), elapsed


def full_p_active(logits: np.ndarray, d: int) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    probs = np.exp(x)
    probs /= probs.sum(axis=-1, keepdims=True)
    return p_active(probs, d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/weights/Qwen__Qwen2.5-0.5B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-fp16", action="store_true", help="very slow on CPU")
    ap.add_argument("--out", default="results/calibration/dtype_sensitivity.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model_dir = (REPO_ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)

    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if args.include_fp16:
        dtypes["float16"] = torch.float16

    report: dict = {"model": args.model, "seed": args.seed, "domains": {}}

    for domain, text in PROBES.items():
        cached: dict[str, np.ndarray] = {}
        timings: dict[str, float] = {}
        for name, dt in dtypes.items():
            cached[name], timings[name] = logits_for(model_dir, dt, text, tok)

        ref = cached["float32"]
        entry: dict = {"n_positions": int(ref.shape[0]), "forward_seconds": timings, "by_dtype": {}}

        for name in dtypes:
            if name == "float32":
                continue
            per_d = {}
            for d in DEPTHS:
                a = full_p_active(ref, d)
                b = full_p_active(cached[name], d)
                nonzero = a > 0
                rel = np.abs(b[a > 0] - a[nonzero]) / a[nonzero]
                low = a < 1e-3
                per_d[str(d)] = {
                    "n_eff_rel_err": float(abs(b.sum() - a.sum()) / a.sum()),
                    "p_active_rel_err_median": float(np.median(rel)) if rel.size else 0.0,
                    "p_active_rel_err_p95": float(np.percentile(rel, 95)) if rel.size else 0.0,
                    "p_active_rel_err_max": float(rel.max()) if rel.size else 0.0,
                    "low_capacity_rel_err_max": (
                        float(np.abs(b[low] - a[low]).max() / max(a[low].min(), 1e-300))
                        if low.any()
                        else None
                    ),
                    "n_low_capacity_positions": int(low.sum()),
                }
            entry["by_dtype"][name] = per_d
        report["domains"][domain] = entry

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(canonical_json(report) + "\n", encoding="utf-8")

    print(f"\n{'domain':10s} {'dtype':9s} {'d':>2s} {'N_eff rel err':>14s} {'p_act med':>10s} {'p_act p95':>10s}")
    for domain, entry in report["domains"].items():
        for name, per_d in entry["by_dtype"].items():
            for d, m in per_d.items():
                print(
                    f"{domain:10s} {name:9s} {d:>2s} {m['n_eff_rel_err']:>14.2%} "
                    f"{m['p_active_rel_err_median']:>10.2%} {m['p_active_rel_err_p95']:>10.2%}"
                )
    print("\nforward seconds:", json.dumps({k: v["forward_seconds"] for k, v in report["domains"].items()}))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
