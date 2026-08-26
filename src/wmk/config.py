from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

__all__ = [
    "WatermarkConfig",
    "PERMITTED_DTYPES",
    "canonical_json",
    "config_hash",
    "tokenizer_hash",
    "provenance",
    "require_provenance",
    "REQUIRED_PROVENANCE_FIELDS",
]

PERMITTED_DTYPES = ("float32", "bfloat16", "float16")


@dataclass(frozen=True)
class WatermarkConfig:
    """A capacity number is only meaningful alongside its full serving tuple, so
    (temperature, top_p, top_k) are hashed into config_hash with the scheme fields."""

    tokenizer_id: str
    tokenizer_hash: str

    prf: str = "hmac-sha256"
    context_depth: int = 4
    tournament_depth: int = 2
    masking: str = "repeated-context"
    masking_window: int = 512
    sampler_position: str = "post-truncation"

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    dtype: str = "float32"
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.prf != "hmac-sha256":
            raise ValueError(f"unsupported prf: {self.prf!r}")
        if self.masking != "repeated-context":
            raise ValueError(f"unsupported masking: {self.masking!r}")
        if self.sampler_position != "post-truncation":
            raise ValueError(f"unsupported sampler_position: {self.sampler_position!r}")
        if self.dtype not in PERMITTED_DTYPES:
            raise ValueError(f"dtype {self.dtype!r} not in {PERMITTED_DTYPES}")
        if self.context_depth < 1:
            raise ValueError("context_depth must be >= 1")
        if self.tournament_depth < 1:
            raise ValueError("tournament_depth must be >= 1")
        if self.masking_window < 1:
            raise ValueError("masking_window must be >= 1")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")

    @property
    def n_candidates(self) -> int:
        return 2 ** self.tournament_depth

    @property
    def config_hash(self) -> str:
        return config_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_(self, **changes: Any) -> "WatermarkConfig":
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WatermarkConfig":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**dict(d))

    @classmethod
    def load(cls, path: str) -> "WatermarkConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(canonical_json(self.to_dict()) + "\n")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(cfg: WatermarkConfig) -> str:
    return hashlib.sha256(canonical_json(cfg.to_dict()).encode("ascii")).hexdigest()


def tokenizer_hash(tokenizer: Any) -> str:
    """Pins the exact token-id assignment the PRF hashes over."""
    vocab = tokenizer.get_vocab()
    h = hashlib.sha256()
    h.update(f"n={len(vocab)}\n".encode("utf-8"))
    for line in sorted(f"{tok}\t{idx}" for tok, idx in vocab.items()):
        h.update(line.encode("utf-8") + b"\n")
    return h.hexdigest()


REQUIRED_PROVENANCE_FIELDS = ("config_hash", "tokenizer_hash", "key_id", "dtype")


def provenance(cfg: WatermarkConfig, key_id: str | int, **extra: Any) -> dict[str, Any]:
    block = {
        "config_hash": cfg.config_hash,
        "tokenizer_hash": cfg.tokenizer_hash,
        "tokenizer_id": cfg.tokenizer_id,
        "key_id": str(key_id),
        "dtype": cfg.dtype,
        "schema_version": cfg.schema_version,
    }
    block.update(extra)
    return block


def require_provenance(record: Mapping[str, Any]) -> None:
    missing = [f for f in REQUIRED_PROVENANCE_FIELDS if not record.get(f)]
    if missing:
        raise ValueError(f"result record missing provenance fields: {missing}")
