"""Temperature / top-k / top-p truncation, applied in that order.

Two entry points: truncate_full for generation, where the whole logit row is in hand,
and truncate_from_cache for the Phase 3 grid, which sees only the cached top-K plus
the full-row log-sum-exp.

The cached path stays exact because renormalizing over a surviving set cancels the
full-row normalizer. What it cannot see is whether the nucleus extends past the cached
K, so it bounds the uncached tail's mass and flags positions where the bound is too
weak to decide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import WatermarkConfig

__all__ = ["truncate_full", "truncate_from_cache", "CachedTruncation"]


@dataclass(frozen=True)
class CachedTruncation:
    """probs: (T, K) float64, renormalized over the surviving set, non-survivors zeroed.

    beyond_mass bounds the renormalized mass of survivors that fall outside the cached
    K; q_min is the smallest surviving renormalized probability. Together they bound
    the capacity error, see wmk.capacity.p_active_bounds. insufficient marks positions
    where the nucleus provably escapes the cache and no bound is available."""

    probs: np.ndarray
    beyond_mass: np.ndarray
    q_min: np.ndarray
    insufficient: np.ndarray
    tail_share_bound: np.ndarray


def _nucleus_keep(sorted_probs: np.ndarray, top_p: float) -> np.ndarray:
    """Smallest descending prefix whose cumulative mass reaches top_p."""
    cum = np.cumsum(sorted_probs, axis=-1)
    keep = cum - sorted_probs < top_p
    keep[..., 0] = True
    return keep


def truncate_full(logits: np.ndarray, cfg: WatermarkConfig) -> np.ndarray:
    """logits: (..., V) full row. Returns (..., V) float64 renormalized probabilities
    with truncated entries exactly zero."""
    scaled = np.asarray(logits, dtype=np.float64) / cfg.temperature
    order = np.argsort(-scaled, axis=-1, kind="stable")
    ordered = np.take_along_axis(scaled, order, axis=-1)

    ordered = ordered - ordered.max(axis=-1, keepdims=True)
    probs = np.exp(ordered)
    probs /= probs.sum(axis=-1, keepdims=True)

    keep = np.ones(probs.shape, dtype=bool)
    if cfg.top_k > 0:
        keep[..., cfg.top_k:] = False
    if cfg.top_p < 1.0:
        keep &= _nucleus_keep(probs, cfg.top_p)

    probs = np.where(keep, probs, 0.0)
    probs /= probs.sum(axis=-1, keepdims=True)

    out = np.empty_like(probs)
    np.put_along_axis(out, order, probs, axis=-1)
    return out


def truncate_from_cache(
    topk_logits: np.ndarray,
    logsumexp_full: np.ndarray,
    cfg: WatermarkConfig,
    vocab_size: int,
) -> CachedTruncation:
    """topk_logits: (T, K) float, descending along K. logsumexp_full: (T,) at T=1.

    Raises for (temperature != 1, top_p == 1, top_k == 0): an untruncated row at a
    shifted temperature needs the full-vocab normalizer, which the cache does not hold.
    """
    logits = np.asarray(topk_logits, dtype=np.float64)
    lse = np.asarray(logsumexp_full, dtype=np.float64)
    n_pos, k = logits.shape
    untruncated = cfg.top_p >= 1.0 and cfg.top_k == 0

    if untruncated and cfg.temperature != 1.0:
        raise NotImplementedError(
            "temperature != 1 with no truncation cannot be recovered from a top-K "
            "cache; the full-vocab normalizer does not cancel. Cache full logits or "
            "pair the temperature with top_p < 1."
        )

    if cfg.temperature > 1.0:
        raise NotImplementedError(
            "temperature > 1 flattens the row, so the uncached tail's mass is no "
            "longer boundable from a top-K cache"
        )

    p1 = np.exp(logits - lse[:, None])
    residual_1 = np.clip(1.0 - p1.sum(axis=-1), 0.0, 1.0)

    scaled = logits / cfg.temperature
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    cached_soft = np.exp(shifted)
    cached_soft /= cached_soft.sum(axis=-1, keepdims=True)

    # exp(l_i/T) <= exp(l_i) * exp(l_K * (1/T - 1)) for l_i <= l_K and T <= 1, which
    # bounds the uncached tail's share without ever seeing the tail.
    inv = 1.0 / cfg.temperature - 1.0
    with np.errstate(divide="ignore"):
        log_tail_num = logits[:, -1] * inv + lse + np.log(residual_1)
    log_cached_den = scaled.max(axis=-1) + np.log(np.exp(shifted).sum(axis=-1))
    tail_share = np.where(
        residual_1 > 0.0,
        np.exp(np.minimum(log_tail_num - log_cached_den, 0.0)),
        0.0,
    )

    keep = np.ones((n_pos, k), dtype=bool)
    if cfg.top_k > 0:
        keep[:, cfg.top_k:] = False

    insufficient = np.zeros(n_pos, dtype=bool)
    if cfg.top_p < 1.0:
        conservative = cached_soft * (1.0 - tail_share)[:, None]
        keep &= _nucleus_keep(conservative, cfg.top_p)
        insufficient = conservative.sum(axis=-1) < cfg.top_p

    if untruncated:
        probs = p1
        beyond = residual_1
        q_min = p1[:, -1]
    else:
        probs = np.where(keep, cached_soft, 0.0)
        totals = probs.sum(axis=-1, keepdims=True)
        probs = np.divide(probs, totals, out=np.zeros_like(probs), where=totals > 0.0)
        # A nucleus touching the last cached slot may continue past it; the cache
        # cannot say how far, so those positions are reported rather than estimated.
        insufficient |= keep[:, -1]
        beyond = np.zeros(n_pos, dtype=np.float64)
        idx = np.clip(keep.sum(axis=-1) - 1, 0, k - 1)
        q_min = np.take_along_axis(probs, idx[:, None], axis=-1)[:, 0]

    return CachedTruncation(
        probs=probs,
        beyond_mass=beyond,
        q_min=q_min,
        insufficient=insufficient,
        tail_share_bound=tail_share,
    )
