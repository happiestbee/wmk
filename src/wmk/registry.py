"""Model registry and weight manifest access. Model selections are data, not code."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

__all__ = ["REPO_ROOT", "ModelEntry", "load_registry", "all_repos", "load_manifest", "sha256_file"]

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data" / "models.json"
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"


@dataclass(frozen=True)
class ModelEntry:
    repo_id: str
    role: str
    group: str
    params: str
    family: str


def load_registry(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or REGISTRY_PATH).read_text(encoding="utf-8"))


def all_repos(registry: dict[str, Any] | None = None, include_optional: bool = False) -> list[ModelEntry]:
    reg = registry if registry is not None else load_registry()
    seen: dict[str, ModelEntry] = {}

    def add(entry: ModelEntry) -> None:
        seen.setdefault(entry.repo_id, entry)

    for pair in reg["architecture_pairs"]:
        for role in ("base", "instruct"):
            add(ModelEntry(pair[role], role, f"pair_{pair['pair']}", pair["params"], pair["family"]))
    for item in reg["scale_series"]:
        for role in ("base", "instruct"):
            add(ModelEntry(item[role], role, "scale_series", item["params"], "qwen2"))
    if include_optional:
        for item in reg["optional_extension"]:
            for role in ("base", "instruct"):
                add(ModelEntry(item[role], role, "optional_7b", item["params"], "qwen2"))
    return list(seen.values())


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    p = path or MANIFEST_PATH
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run scripts/00_setup.py first")
    return json.loads(p.read_text(encoding="utf-8"))


def iter_weight_files(root: Path) -> Iterator[Path]:
    for pattern in ("*.safetensors", "*.json", "*.txt", "*.model"):
        yield from sorted(root.rglob(pattern))
