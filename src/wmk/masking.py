"""Repeated-context masking, shared by the Phase 3 capacity scan and the Phase 5
scorer. N_eff and the detector must agree on the unmasked set or every F1 number is
optimistic; sharing this function is how they agree."""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["scorable_mask", "masked_fraction"]


def scorable_mask(
    token_ids: Sequence[int],
    context_depth: int,
    masking_window: int,
) -> np.ndarray:
    """Bool array over positions: True where the position carries watermark signal.

    A position is dropped when it has no full context window, or when its
    context_depth-token context recurred within the preceding masking_window
    positions. Last-seen is updated on every position, masked or not."""
    ids = np.asarray(token_ids, dtype=np.int64)
    n = int(ids.shape[0])
    mask = np.zeros(n, dtype=bool)
    if n <= context_depth:
        return mask

    last_seen: dict[bytes, int] = {}
    raw = ids.astype(np.uint32).tobytes()
    stride = 4
    for t in range(context_depth, n):
        ctx = raw[(t - context_depth) * stride: t * stride]
        prev = last_seen.get(ctx)
        if prev is None or (t - prev) > masking_window:
            mask[t] = True
        last_seen[ctx] = t
    return mask


def masked_fraction(mask: np.ndarray, context_depth: int) -> float:
    """Share of full-context positions that masking dropped, so the no-context
    prefix does not inflate the reported fraction."""
    eligible = int(mask.shape[0]) - context_depth
    if eligible <= 0:
        return 0.0
    return float(eligible - int(mask.sum())) / float(eligible)
