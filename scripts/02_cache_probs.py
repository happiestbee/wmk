#!/usr/bin/env python3
"""Phase 2: one forward pass per document, ever.

Each pass evaluates p_active for the whole serving grid while the full distribution is
in hand, and caches those values plus the token ids. Phase 3 then sweeps
(T, top_p, top_k, d) by indexing the cache and masking_window from the token ids, with
no second model pass.

Row i of the per-position arrays is the distribution that emitted token positions[i] of
its document -- logits index positions[i]-1. The alignment is stored rather than
implied, because an off-by-one here is invisible downstream.

    python scripts/02_cache_probs.py --model Qwen/Qwen2.5-0.5B --domain prose
    python scripts/02_cache_probs.py --all
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

from wmk.config import canonical_json, tokenizer_hash
from wmk.grid import load_grid
from wmk.registry import REPO_ROOT, all_repos, load_manifest
from wmk.scan import evaluate_logits

PROBS_DIR = REPO_ROOT / "data" / "probs"
CORPORA_DIR = REPO_ROOT / "data" / "corpora"
CHECKPOINT_EVERY = 50

ARRAY_KEYS = (
    "p_active",
    "entropy",
    "top1_prob",
    "p_active_slack",
    "nucleus_size",
    "kept_mass",
    "nucleus_overflow",
    "topk_logits",
    "topk_ids",
    "lse",
    "positions",
    "doc_index",
    "token_ids",
)


def cache_path(repo_id: str, domain: str) -> Path:
    return PROBS_DIR / f"{repo_id.replace('/', '__')}__{domain}.npz"


def load_docs(domain: str) -> list[dict]:
    path = CORPORA_DIR / f"{domain}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _save(path: Path, state: dict, meta: dict) -> None:
    arrays = {}
    for key in ARRAY_KEYS:
        parts = state[key]
        arrays[key] = np.concatenate(parts) if parts else np.zeros(0)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            doc_offsets=np.array(state["doc_offsets"], np.int64),
            doc_ids=np.array(state["doc_ids"], dtype=object),
            meta=np.array(canonical_json(meta), dtype=object),
            **arrays,
        )
    tmp.replace(path)


def _empty_state() -> dict:
    state: dict = {k: [] for k in ARRAY_KEYS}
    state["doc_offsets"] = [0]
    state["doc_ids"] = []
    return state


def cache_model_domain(
    repo_id: str,
    local_dir: Path,
    domain: str,
    dtype: str,
    limit: int | None = None,
    chunk: int = 128,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    grid = load_grid()
    out = cache_path(repo_id, domain)
    docs = load_docs(domain)
    if limit:
        docs = docs[:limit]

    state = _empty_state()
    done: set[str] = set()
    grid_signature = canonical_json(
        {
            "serving": [s.__dict__ for s in grid.serving_configs],
            "depths": list(grid.tournament_depths),
            "top_k_cache": grid.top_k_cache,
            "nucleus_k": grid.nucleus_k,
        }
    )

    if out.exists():
        with np.load(out, allow_pickle=True) as z:
            prior = json.loads(str(z["meta"]))
            if prior.get("grid_signature") == grid_signature and prior.get("dtype") == dtype:
                for key in ARRAY_KEYS:
                    arr = z[key]
                    state[key] = [arr] if arr.shape[0] else []
                state["doc_offsets"] = list(z["doc_offsets"])
                state["doc_ids"] = [str(v) for v in z["doc_ids"]]
                done = set(state["doc_ids"])
                print(f"  resuming: {len(done)} docs cached")
            else:
                print("  cache exists for a different grid or dtype; rebuilding")

    todo = [d for d in docs if d["doc_id"] not in done]
    if not todo:
        with np.load(out, allow_pickle=True) as z:
            print(f"  complete ({len(done)} docs)")
            return json.loads(str(z["meta"]))

    tok = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForCausalLM.from_pretrained(local_dir, dtype=getattr(torch, dtype))
    model.eval()

    meta = {
        "repo_id": repo_id,
        "domain": domain,
        "dtype": dtype,
        "tokenizer_id": repo_id,
        "tokenizer_hash": tokenizer_hash(tok),
        "vocab_size": int(model.config.vocab_size),
        "grid_signature": grid_signature,
        "serving_labels": [s.label for s in grid.serving_configs],
        "tournament_depths": list(grid.tournament_depths),
        "row_alignment": "row i emitted token positions[i]; logits index positions[i]-1",
        "torch_threads": torch.get_num_threads(),
    }

    t_start = time.perf_counter()
    n_tokens = 0
    for i, doc in enumerate(todo):
        ids = tok(doc["text"], add_special_tokens=False, return_tensors="pt").input_ids
        n = int(ids.shape[1])
        if n < 2:
            continue
        with torch.no_grad():
            logits = model(ids).logits[0].float()

        dist = logits[:-1]
        evaluated = evaluate_logits(dist, grid, chunk=chunk)
        top = torch.topk(dist, grid.top_k_cache, dim=-1)

        for key, value in evaluated.items():
            state[key].append(value)
        state["topk_logits"].append(top.values.numpy().astype(np.float32))
        state["topk_ids"].append(top.indices.numpy().astype(np.int32))
        state["lse"].append(torch.logsumexp(dist, dim=-1).numpy().astype(np.float32))
        state["positions"].append(np.arange(1, n, dtype=np.int32))
        state["doc_index"].append(np.full(n - 1, len(state["doc_ids"]), dtype=np.int32))
        state["token_ids"].append(ids[0].numpy().astype(np.int32))
        state["doc_offsets"].append(state["doc_offsets"][-1] + n)
        state["doc_ids"].append(doc["doc_id"])
        n_tokens += n

        if (i + 1) % CHECKPOINT_EVERY == 0 or i + 1 == len(todo):
            _save(out, state, meta)
            rate = n_tokens / (time.perf_counter() - t_start)
            eta = (len(todo) - i - 1) * (time.perf_counter() - t_start) / (i + 1) / 60
            print(f"  {i + 1}/{len(todo)} docs, {n_tokens} tok, {rate:.0f} tok/s, eta {eta:.0f}m", flush=True)

    _save(out, state, meta)
    del model
    meta["n_docs"] = len(state["doc_ids"])
    meta["seconds"] = round(time.perf_counter() - t_start, 1)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--domains", nargs="*", default=["prose", "factual", "code"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=128)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    PROBS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    if args.model:
        repos = [args.model]
    elif args.all:
        repos = [e.repo_id for e in all_repos()]
    else:
        ap.error("pass --model or --all")

    domains = [args.domain] if args.domain else args.domains
    for repo_id in repos:
        info = manifest["models"].get(repo_id)
        if not info:
            print(f"SKIP {repo_id}: not in MANIFEST.json", file=sys.stderr)
            continue
        for domain in domains:
            print(f"{repo_id} / {domain}", flush=True)
            cache_model_domain(
                repo_id, REPO_ROOT / info["local_dir"], domain, args.dtype,
                limit=args.limit, chunk=args.chunk,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
