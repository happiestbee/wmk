#!/usr/bin/env python3
"""Phase 0.5: the whole pipeline on one model, one domain, ten documents.

PRF -> cached probs -> p_active/N_eff -> watermarked generation -> scoring, compared
against an unwatermarked control and against a wrong key. Catches integration bugs
here rather than in Phase 3 or 4.

Gate 0.5: watermarked text scores measurably above unwatermarked, and N_eff is
non-zero and below the token count.

    python scripts/00b_smoke_test.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmk.config import WatermarkConfig, canonical_json, tokenizer_hash
from wmk.generate import generate
from wmk.keys import null_key
from wmk.masking import masked_fraction, scorable_mask
from wmk.registry import REPO_ROOT, load_manifest
from wmk.score import score_token_ids
from wmk.grid import load_grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

cache_mod = import_module("02_cache_probs")


def capacity_from_cache(repo_id: str, domain: str, cfg: WatermarkConfig) -> dict:
    grid = load_grid()
    serving = grid.reference_serving
    si = grid.serving_configs.index(serving)
    di = grid.tournament_depths.index(cfg.tournament_depth)

    path = cache_mod.cache_path(repo_id, domain)
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["meta"]))
        pa = z["p_active"][:, si, di]
        entropy, nucleus = z["entropy"], z["nucleus_size"][:, si]
        positions, doc_index = z["positions"], z["doc_index"]
        token_ids, offsets = z["token_ids"], z["doc_offsets"]
        doc_ids = [str(v) for v in z["doc_ids"]]

    per_doc = []
    for doc_i, doc_id in enumerate(doc_ids):
        rows = doc_index == doc_i
        doc_tokens = token_ids[offsets[doc_i]: offsets[doc_i + 1]]
        mask = scorable_mask(doc_tokens, cfg.context_depth, cfg.masking_window)
        # positions are emitted-token indices; align p_active rows to the mask.
        row_positions = positions[rows]
        aligned = np.zeros(doc_tokens.shape[0], dtype=np.float64)
        aligned[row_positions] = pa[rows]
        per_doc.append(
            {
                "doc_id": doc_id,
                "n_tokens": int(doc_tokens.shape[0]),
                "n_scored": int(mask.sum()),
                "masked_fraction": masked_fraction(mask, cfg.context_depth),
                "n_eff": float(aligned[mask].sum()),
                "mean_p_active": float(aligned[mask].mean()) if mask.any() else 0.0,
            }
        )

    return {
        "meta": meta,
        "per_doc": per_doc,
        "mean_entropy_nats": float(entropy.mean()),
        "median_nucleus_size": float(np.median(nucleus)),
        "serving_label": serving.label,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--domain", default="prose")
    ap.add_argument("--n-docs", type=int, default=10)
    ap.add_argument("--new-tokens", type=int, default=200)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    manifest = load_manifest()
    info = manifest["models"][args.model]
    local_dir = REPO_ROOT / info["local_dir"]

    print(f"== Phase 0.5 smoke test: {args.model} / {args.domain} / {args.n_docs} docs")

    print("\n[1/4] caching probabilities")
    cache_mod.cache_model_domain(
        args.model, local_dir, args.domain, args.dtype, limit=args.n_docs
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(local_dir)
    cfg = WatermarkConfig(
        tokenizer_id=args.model, tokenizer_hash=tokenizer_hash(tok), dtype=args.dtype
    )
    print(f"      config_hash={cfg.config_hash[:16]} d={cfg.tournament_depth} H={cfg.context_depth}")

    print("\n[2/4] capacity from cache")
    cap = capacity_from_cache(args.model, args.domain, cfg)
    n_eff = np.array([d["n_eff"] for d in cap["per_doc"]])
    n_tokens = np.array([d["n_tokens"] for d in cap["per_doc"]])
    mean_pa = np.array([d["mean_p_active"] for d in cap["per_doc"]])
    print(f"      N_eff mean={n_eff.mean():.1f} range=[{n_eff.min():.1f}, {n_eff.max():.1f}]")
    print(f"      tokens mean={n_tokens.mean():.0f}  mean p_active={mean_pa.mean():.4f}")
    print(f"      masked fraction={np.mean([d['masked_fraction'] for d in cap['per_doc']]):.2%}")
    print(f"      mean entropy={cap['mean_entropy_nats']:.3f} nats, median nucleus={cap['median_nucleus_size']:.0f} ({cap['serving_label']})")

    print(f"\n[3/4] generating {args.n_docs} watermarked + {args.n_docs} control")
    model = AutoModelForCausalLM.from_pretrained(local_dir, dtype=getattr(torch, args.dtype))
    model.eval()
    docs = cache_mod.load_docs(args.domain)[: args.n_docs]
    key = null_key(0)
    wrong_key = null_key(1)

    runs = []
    t0 = time.perf_counter()
    for i, doc in enumerate(docs):
        prompt = tok.decode(tok(doc["text"], add_special_tokens=False).input_ids[:32])
        for label, k in (("watermarked", key), ("control", None)):
            out = generate(model, tok, prompt, cfg, k, args.new_tokens, seed=args.seed + i)
            runs.append({"doc_id": doc["doc_id"], "label": label, **out})
        print(f"      {i + 1}/{len(docs)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    print("\n[4/4] scoring")
    rows = []
    for run in runs:
        for key_label, k, kid in (("true", key, "0"), ("wrong", wrong_key, "1")):
            res = score_token_ids(run["token_ids"], cfg, k, key_id=kid)
            rows.append(
                {
                    "doc_id": run["doc_id"],
                    "text_label": run["label"],
                    "key_label": key_label,
                    "n_tokens": len(run["token_ids"]),
                    "n_scored": res.n_scored,
                    "mean_g": res.mean_g,
                    "z_null": res.z_null,
                }
            )

    def cell(text_label: str, key_label: str) -> tuple[float, float]:
        z = [r["z_null"] for r in rows if r["text_label"] == text_label and r["key_label"] == key_label]
        g = [r["mean_g"] for r in rows if r["text_label"] == text_label and r["key_label"] == key_label]
        return float(np.mean(g)), float(np.mean(z))

    print(f"\n      {'text':13s} {'key':6s} {'mean_g':>8s} {'mean z':>8s}")
    cells = {}
    for tl in ("watermarked", "control"):
        for kl in ("true", "wrong"):
            g, z = cell(tl, kl)
            cells[(tl, kl)] = (g, z)
            print(f"      {tl:13s} {kl:6s} {g:>8.4f} {z:>8.2f}")

    wm_z = cells[("watermarked", "true")][1]
    ctrl_z = cells[("control", "true")][1]
    wrong_z = cells[("watermarked", "wrong")][1]

    checks = {
        "watermarked_above_control": bool(wm_z > ctrl_z + 2.0),
        "wrong_key_looks_null": bool(abs(wrong_z) < 3.0),
        "n_eff_nonzero": bool(n_eff.min() > 0),
        "n_eff_below_token_count": bool(np.all(n_eff < n_tokens)),
    }
    passed = all(checks.values())

    print("\n== Gate 0.5")
    for name, ok in checks.items():
        print(f"   {'PASS' if ok else 'FAIL'}  {name}")
    print(f"   watermarked z={wm_z:.2f}  control z={ctrl_z:.2f}  wrong-key z={wrong_z:.2f}")

    report = {
        "model": args.model,
        "domain": args.domain,
        "seed": args.seed,
        "config": cfg.to_dict(),
        "config_hash": cfg.config_hash,
        "capacity": {
            "n_eff_mean": float(n_eff.mean()),
            "n_tokens_mean": float(n_tokens.mean()),
            "mean_p_active": float(mean_pa.mean()),
            "mean_entropy_nats": cap["mean_entropy_nats"],
            "median_nucleus_size": cap["median_nucleus_size"],
            "per_doc": cap["per_doc"],
        },
        "score_cells": {f"{a}|{b}": {"mean_g": v[0], "mean_z": v[1]} for (a, b), v in cells.items()},
        "rows": rows,
        "checks": checks,
        "passed": passed,
    }
    out = REPO_ROOT / "results" / "calibration" / "phase_0_5_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
