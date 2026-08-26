from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .masking import scorable_mask

__all__ = ["p_active", "p_active_bounds", "n_eff", "n_eff_min", "tokens_for_n_eff"]


def p_active(probs: np.ndarray, d: int) -> np.ndarray:
    """probs must be post-truncation and renormalized; rows summing to anything but
    1.0 give silently wrong results. Returns (T,).

    1 - sum_i p_i**(2**d) is exact, not an approximation: it is the probability that
    the 2**d independent draws do not all collide on one token, and collision events
    are mutually exclusive across tokens."""
    if d < 1:
        raise ValueError(f"tournament_depth must be >= 1, got {d}")
    # float64 is required for cancellation, not underflow: at a near-deterministic
    # position sum_i p_i**(2**d) approaches 1, and float32 leaves ~eps/p_active
    # relative error in the difference -- 1.3% at p_active 2e-6, which is the code
    # and factual regime F1 reports on.
    p = np.asarray(probs, dtype=np.float64)
    return 1.0 - np.sum(p ** (2 ** d), axis=-1)


def p_active_bounds(
    probs: np.ndarray,
    beyond_mass: np.ndarray,
    q_min: np.ndarray,
    d: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided bound on p_active when probs covers only the cached top-K.

    Survivors outside the cache each carry at most q_min, so their contribution to
    sum_i q_i**(2**d) lies in [0, beyond_mass * q_min**(2**d - 1)]. Returns (lo, hi);
    hi is the value p_active would report from the cache alone."""
    power = 2 ** d
    hi = p_active(probs, d)
    slack = np.asarray(beyond_mass, dtype=np.float64) * np.asarray(
        q_min, dtype=np.float64
    ) ** (power - 1)
    return hi - slack, hi


def n_eff(p_active_values: np.ndarray, mask: np.ndarray) -> float:
    """Sums over unmasked positions only. Including masked positions overstates the
    detector's input and every downstream F1 number with it."""
    pa = np.asarray(p_active_values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if pa.shape != m.shape:
        raise ValueError(f"shape mismatch: p_active {pa.shape} vs mask {m.shape}")
    return float(pa[m].sum())


def n_eff_min(delta: float, d: int, alpha: float, beta: float = 0.9) -> float:
    """Effective positions needed for power beta at false-positive rate alpha, given
    the per-observation bias delta. Analytic binomial null: a sanity check only, since
    positions correlate through shared context (§2.3)."""
    if not 0.0 < delta:
        raise ValueError("delta must be > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    z = norm.ppf(1.0 - alpha) + norm.ppf(beta)
    return float(z * z / (4.0 * delta * delta * d))


def tokens_for_n_eff(target_n_eff: float, mean_p_active: float, masked_fraction: float) -> float:
    """Converts an N_eff requirement into model-written tokens for a domain whose
    unmasked positions carry mean_p_active each."""
    if mean_p_active <= 0.0:
        return float("inf")
    if not 0.0 <= masked_fraction < 1.0:
        raise ValueError("masked_fraction must be in [0, 1)")
    return float(target_n_eff / (mean_p_active * (1.0 - masked_fraction)))


def document_n_eff(
    probs: np.ndarray,
    token_ids: np.ndarray,
    context_depth: int,
    masking_window: int,
    d: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns (n_eff, p_active per position, scorable mask). probs rows must align
    with the *predicted* token at each position."""
    mask = scorable_mask(token_ids, context_depth, masking_window)
    pa = p_active(probs, d)
    if pa.shape[0] != mask.shape[0]:
        raise ValueError(f"p_active has {pa.shape[0]} positions, mask has {mask.shape[0]}")
    return n_eff(pa, mask), pa, mask
