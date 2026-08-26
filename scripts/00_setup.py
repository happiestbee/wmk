#!/usr/bin/env python3
"""Phase 0: probe the host, verify every model repo is reachable anonymously,
stage weights, and hash them into data/MANIFEST.json.

    python scripts/00_setup.py --probe-only
    python scripts/00_setup.py --download --pairs A
    python scripts/00_setup.py --download --all
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmk.config import canonical_json
from wmk.registry import (
    MANIFEST_PATH,
    REPO_ROOT,
    ModelEntry,
    all_repos,
    load_registry,
    sha256_file,
)

WEIGHTS_DIR = REPO_ROOT / "data" / "weights"
LOGS_DIR = REPO_ROOT / "logs"
HF_API = "https://huggingface.co/api/models/"


def probe_host() -> dict:
    dev = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "mkldnn": torch.backends.mkldnn.is_available(),
    }
    if dev["cuda_available"]:
        dev["gpu_name"] = torch.cuda.get_device_name(0)
        dev["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        dev["device"] = "cuda"
    else:
        dev["device"] = "cpu"

    dev["dtype_matmul_ms"] = _time_dtypes()
    return dev


def _time_dtypes(size: int = 2048, reps: int = 5) -> dict[str, float]:
    """C8 requires recording the dtype; on a CPU host it also has to justify it, so
    the cost of each permitted dtype is measured rather than assumed."""
    out = {}
    for name, dt in (("float32", torch.float32), ("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        a = torch.randn(512, size, dtype=dt)
        b = torch.randn(size, size, dtype=dt)
        t0 = time.perf_counter()
        for _ in range(reps):
            a @ b
        out[name] = round((time.perf_counter() - t0) / reps * 1e3, 2)
    return out


def check_reachable(repo_id: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(HF_API + repo_id, headers={"User-Agent": "wmk/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            meta = json.loads(resp.read())
        return {
            "repo_id": repo_id,
            "status": resp.status,
            "gated": meta.get("gated", False),
            "sha": meta.get("sha"),
            "ok": resp.status == 200 and not meta.get("gated", False),
        }
    except urllib.error.HTTPError as exc:
        return {"repo_id": repo_id, "status": exc.code, "gated": None, "sha": None, "ok": False}
    except Exception as exc:
        return {"repo_id": repo_id, "status": None, "error": str(exc), "ok": False}


def stage_weights(entry: ModelEntry) -> dict:
    from huggingface_hub import snapshot_download

    target = WEIGHTS_DIR / entry.repo_id.replace("/", "__")
    local = snapshot_download(
        repo_id=entry.repo_id,
        local_dir=target,
        # TinyLlama_v1.1 publishes no safetensors; torch.load defaults to
        # weights_only=True, so the pickle carries no execution risk.
        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.model", "*.tiktoken"],
        max_workers=4,
    )
    root = Path(local)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            files[str(path.relative_to(root))] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return {
        "repo_id": entry.repo_id,
        "role": entry.role,
        "group": entry.group,
        "params": entry.params,
        "family": entry.family,
        "local_dir": str(root.relative_to(REPO_ROOT)),
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "files": files,
    }


def smoke_forward(repo_dir: Path, dtype: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wmk.config import tokenizer_hash

    tok = AutoTokenizer.from_pretrained(repo_dir)
    model = AutoModelForCausalLM.from_pretrained(repo_dir, dtype=getattr(torch, dtype))
    model.eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(ids).logits
    elapsed = time.perf_counter() - t0
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    top = torch.topk(probs, 3)
    return {
        "tokenizer_hash": tokenizer_hash(tok),
        "vocab_size": int(logits.shape[-1]),
        "n_positions": int(ids.shape[1]),
        "forward_seconds": round(elapsed, 3),
        "top3": [
            {"token": tok.decode([int(i)]), "p": round(float(p), 4)}
            for p, i in zip(top.values, top.indices)
        ],
        "finite": bool(torch.isfinite(logits).all()),
        "dtype": dtype,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--all", action="store_true", help="include the optional 7B extension")
    ap.add_argument("--pairs", nargs="*", default=None, help="restrict to these pair letters")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    host = probe_host()
    print(f"device={host['device']} threads={host['torch_threads']} capability={host['cpu_capability']}")
    print(f"matmul ms by dtype: {host['dtype_matmul_ms']}")

    entries = all_repos(registry, include_optional=args.all)
    if args.pairs:
        wanted = {f"pair_{p.upper()}" for p in args.pairs}
        entries = [e for e in entries if e.group in wanted]

    reach = [check_reachable(e.repo_id) for e in entries]
    blocked = [r for r in reach if not r["ok"]]
    for r in reach:
        print(f"  {'ok ' if r['ok'] else 'BLOCKED'} {r['repo_id']} status={r['status']} gated={r['gated']}")
    if blocked:
        print(f"\n{len(blocked)} repo(s) unreachable. Fallbacks: {registry['fallbacks']}", file=sys.stderr)

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "host": host,
        "registry_selections": [e.__dict__ for e in entries],
        "reachability": reach,
        "models": {},
    }
    if MANIFEST_PATH.exists():
        manifest["models"] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("models", {})

    if args.probe_only:
        _write(manifest)
        return 1 if blocked else 0

    if args.download:
        for entry in entries:
            if entry.repo_id in manifest["models"]:
                print(f"skip {entry.repo_id} (already staged)")
                continue
            print(f"staging {entry.repo_id} ...", flush=True)
            manifest["models"][entry.repo_id] = stage_weights(entry)
            _write(manifest)
            gb = manifest["models"][entry.repo_id]["total_bytes"] / 1e9
            print(f"  {manifest['models'][entry.repo_id]['n_files']} files, {gb:.2f} GB")

    if args.smoke:
        for entry in entries:
            info = manifest["models"].get(entry.repo_id)
            if not info:
                continue
            if info.get("smoke", {}).get("finite"):
                continue
            try:
                result = smoke_forward(REPO_ROOT / info["local_dir"], args.dtype)
                print(f"  smoke {entry.repo_id}: {result['top3'][0]} ({result['forward_seconds']}s)")
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}", "finite": False, "dtype": args.dtype}
                print(f"  SMOKE FAILED {entry.repo_id}: {result['error']}", file=sys.stderr)
            manifest["models"][entry.repo_id]["smoke"] = result
            _write(manifest)

    _write(manifest)
    return 1 if blocked else 0


def _write(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(canonical_json(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
