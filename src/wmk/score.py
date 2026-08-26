"""Detector: one O(N) pass accumulating g-values over unmasked positions.

Per-position g-values never leave this module. That vector is a removal oracle --
anyone holding it can edit exactly the positions carrying signal -- so the boundary
exposes aggregates only, and window sums are floored at MIN_WINDOW to keep a
window-of-one from reconstructing it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import WatermarkConfig, provenance, tokenizer_hash
from .masking import masked_fraction, scorable_mask
from .prf import PRF

__all__ = ["ScoreResult", "score_token_ids", "score_text", "MIN_WINDOW"]

MIN_WINDOW = 32


class TokenizerMismatch(ValueError):
    """A differing token-id assignment makes the g-value sequence meaningless rather
    than merely noisy, so this is an error and never a low score."""


@dataclass(frozen=True)
class ScoreResult:
    g_sum: int
    n_scored: int
    n_observations: int
    n_tokens: int
    masked_fraction: float
    mean_g: float
    z_null: float
    n_eff: float | None = None
    window_sums: np.ndarray | None = None
    window_counts: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        rec = {
            "g_sum": self.g_sum,
            "n_scored": self.n_scored,
            "n_observations": self.n_observations,
            "n_tokens": self.n_tokens,
            "masked_fraction": self.masked_fraction,
            "mean_g": self.mean_g,
            "z_null": self.z_null,
            "n_eff": self.n_eff,
        }
        rec.update(self.provenance)
        return rec


def score_token_ids(
    token_ids: Sequence[int],
    cfg: WatermarkConfig,
    key: bytes,
    key_id: str | int = "0",
    p_active_values: np.ndarray | None = None,
    window: int | None = None,
    stride: int | None = None,
) -> ScoreResult:
    """p_active_values, when supplied, must align with token_ids position-for-position;
    n_eff then sums it over exactly the unmasked set the detector used."""
    ids = np.asarray(token_ids, dtype=np.int64)
    n = int(ids.shape[0])
    prf = PRF(key)
    d = cfg.tournament_depth

    mask = scorable_mask(ids, cfg.context_depth, cfg.masking_window)
    scored_positions = np.flatnonzero(mask)

    g_per_position = np.zeros(scored_positions.shape[0], dtype=np.int32)
    for i, t in enumerate(scored_positions):
        context = tuple(int(v) for v in ids[t - cfg.context_depth: t])
        g_per_position[i] = int(prf.bits(context, int(ids[t]), d).sum())

    g_sum = int(g_per_position.sum())
    n_scored = int(scored_positions.shape[0])
    n_obs = n_scored * d
    mean_g = (g_sum / n_obs) if n_obs else 0.0
    z_null = ((g_sum - n_obs / 2.0) / math.sqrt(n_obs / 4.0)) if n_obs else 0.0

    n_eff = None
    if p_active_values is not None:
        pa = np.asarray(p_active_values, dtype=np.float64)
        if pa.shape[0] != n:
            raise ValueError(f"p_active has {pa.shape[0]} positions, token_ids has {n}")
        n_eff = float(pa[mask].sum())

    window_sums = window_counts = None
    if window is not None and n_scored:
        if window < MIN_WINDOW:
            raise ValueError(f"window must be >= {MIN_WINDOW} to keep aggregates aggregate")
        step = stride or window
        starts = range(0, max(1, n_scored - window + 1), step)
        window_sums = np.array([g_per_position[s: s + window].sum() for s in starts], dtype=np.int64)
        window_counts = np.array([min(window, n_scored - s) * d for s in starts], dtype=np.int64)

    return ScoreResult(
        g_sum=g_sum,
        n_scored=n_scored,
        n_observations=n_obs,
        n_tokens=n,
        masked_fraction=masked_fraction(mask, cfg.context_depth),
        mean_g=mean_g,
        z_null=z_null,
        n_eff=n_eff,
        window_sums=window_sums,
        window_counts=window_counts,
        provenance=provenance(cfg, key_id),
    )


def score_text(
    text: str,
    tokenizer: Any,
    cfg: WatermarkConfig,
    key: bytes,
    key_id: str | int = "0",
    **kwargs: Any,
) -> ScoreResult:
    actual = tokenizer_hash(tokenizer)
    if actual != cfg.tokenizer_hash:
        raise TokenizerMismatch(
            f"tokenizer hash {actual[:12]} does not match config {cfg.tokenizer_hash[:12]} "
            f"({cfg.tokenizer_id})"
        )
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return score_token_ids(ids, cfg, key, key_id=key_id, **kwargs)
