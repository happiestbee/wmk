"""The Phase 3 sweep grid, as data.

p_active is evaluated for every serving config during the single forward pass rather
than reconstructed afterwards from a top-K cache. Measured nucleus sizes make the
reconstruction route unsound: at (T=1.0, top_p=0.95) on prose, 23% of positions have a
0.95-nucleus wider than 256 tokens and 6.9% wider than 1024, so no practical top-K
cache can recover the truncated distribution. Evaluating in-pass is exact, and costs
less on disk than caching enough logits to approximate it.

The trade is that a serving config absent from this grid requires a new forward pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import WatermarkConfig
from .registry import REPO_ROOT

__all__ = ["Grid", "ServingConfig", "load_grid"]

GRID_PATH = REPO_ROOT / "data" / "grid.json"


@dataclass(frozen=True)
class ServingConfig:
    temperature: float
    top_p: float
    top_k: int

    @property
    def label(self) -> str:
        return f"T{self.temperature:g}_p{self.top_p:g}_k{self.top_k:d}"


@dataclass(frozen=True)
class Grid:
    serving_configs: tuple[ServingConfig, ...]
    tournament_depths: tuple[int, ...]
    masking_windows: tuple[int, ...]
    reference: dict[str, Any]
    top_k_cache: int
    nucleus_k: int

    def config_for(
        self,
        tokenizer_id: str,
        tokenizer_hash: str,
        serving: ServingConfig,
        tournament_depth: int,
        masking_window: int,
        dtype: str,
    ) -> WatermarkConfig:
        return WatermarkConfig(
            tokenizer_id=tokenizer_id,
            tokenizer_hash=tokenizer_hash,
            tournament_depth=tournament_depth,
            masking_window=masking_window,
            temperature=serving.temperature,
            top_p=serving.top_p,
            top_k=serving.top_k,
            dtype=dtype,
        )

    @property
    def reference_serving(self) -> ServingConfig:
        r = self.reference
        return ServingConfig(r["temperature"], r["top_p"], r["top_k"])


def load_grid(path: Path | None = None) -> Grid:
    raw = json.loads((path or GRID_PATH).read_text(encoding="utf-8"))
    return Grid(
        serving_configs=tuple(ServingConfig(**s) for s in raw["serving_configs"]),
        tournament_depths=tuple(raw["tournament_depths"]),
        masking_windows=tuple(raw["masking_windows"]),
        reference=dict(raw["reference_config"]),
        top_k_cache=int(raw["top_k_cache"]),
        nucleus_k=int(raw["nucleus_k"]),
    )
