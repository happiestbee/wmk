"""In-pass evaluation of the capacity grid over a block of full logit rows.

Evaluating every serving config while the full distribution is in hand makes the
truncated p_active exact. Reconstructing it later from a top-K cache does not work at
any practical K -- see wmk.grid for the measured nucleus sizes.

Work happens on a top-K slice, but the normalizer is always the exact full-vocab
log-sum-exp at the config's temperature, so nucleus boundaries are true cumulative
mass rather than slice-relative. Positions whose nucleus fills the slice are counted in
`nucleus_overflow` instead of being silently truncated.
"""

from __future__ import annotations

import numpy as np
import torch

from .grid import Grid

__all__ = ["evaluate_logits", "GRID_FIELDS"]

GRID_FIELDS = (
    "p_active",
    "p_active_slack",
    "entropy",
    "top1_prob",
    "nucleus_size",
    "kept_mass",
    "nucleus_overflow",
)


def evaluate_logits(logits: torch.Tensor, grid: Grid, chunk: int = 128) -> dict[str, np.ndarray]:
    """logits: (T, V) full rows, one per position. p_active is (T, n_serving, n_depths).

    p_active_slack bounds the contribution of survivors outside the slice, so a
    published value carries its own error bar rather than an assumption."""
    n_pos, vocab = int(logits.shape[0]), int(logits.shape[1])
    n_serving = len(grid.serving_configs)
    depths = tuple(grid.tournament_depths)
    k = min(grid.nucleus_k, vocab)

    out = {
        "p_active": np.empty((n_pos, n_serving, len(depths)), dtype=np.float64),
        "p_active_slack": np.zeros((n_pos, n_serving, len(depths)), dtype=np.float64),
        "entropy": np.empty(n_pos, dtype=np.float32),
        "top1_prob": np.empty(n_pos, dtype=np.float32),
        "nucleus_size": np.empty((n_pos, n_serving), dtype=np.int32),
        "kept_mass": np.empty((n_pos, n_serving), dtype=np.float32),
        "nucleus_overflow": np.zeros((n_pos, n_serving), dtype=bool),
    }

    temperatures = sorted({s.temperature for s in grid.serving_configs})

    for lo in range(0, n_pos, chunk):
        hi = min(lo + chunk, n_pos)
        block = logits[lo:hi].to(torch.float32)

        top_vals = torch.topk(block, k, dim=-1, sorted=True).values.to(torch.float64)

        base = torch.softmax(block, dim=-1)
        out["entropy"][lo:hi] = (
            -(base * torch.log(base.clamp_min(1e-45))).sum(-1)
        ).to(torch.float32).numpy()
        out["top1_prob"][lo:hi] = base.max(dim=-1).values.numpy()
        del base

        per_temp = {}
        for temp in temperatures:
            scaled_full = block if temp == 1.0 else block / temp
            lse = torch.logsumexp(scaled_full.to(torch.float64), dim=-1, keepdim=True)
            per_temp[temp] = torch.exp(top_vals / temp - lse)

        for si, serving in enumerate(grid.serving_configs):
            p = per_temp[serving.temperature]

            if serving.top_p >= 1.0 and serving.top_k == 0:
                # Nothing is truncated, so the whole vocab tail past the slice edge
                # survives and its contribution is the slack.
                q = p
                kept = torch.ones_like(p, dtype=torch.bool)
                mass = p.sum(dim=-1, keepdim=True)
                overflow = torch.zeros(p.shape[0], dtype=torch.bool)
                beyond = (1.0 - mass[:, 0]).clamp_min(0.0)
            else:
                cum = p.cumsum(dim=-1)
                kept = (cum - p) < serving.top_p
                kept[:, 0] = True
                if serving.top_k:
                    kept[:, serving.top_k:] = False
                overflow = kept[:, -1].clone()
                masked = torch.where(kept, p, torch.zeros((), dtype=p.dtype))
                mass = masked.sum(dim=-1, keepdim=True)
                q = masked / mass
                # Mass truncation removed is not a surviving candidate, so it is not
                # error. Only survivors past the slice edge are, and only where the
                # nucleus reached that edge.
                beyond = torch.where(
                    overflow, (1.0 - mass[:, 0]).clamp_min(0.0), torch.zeros((), dtype=p.dtype)
                )

            out["nucleus_size"][lo:hi, si] = kept.sum(dim=-1).to(torch.int32).numpy()
            out["kept_mass"][lo:hi, si] = mass[:, 0].to(torch.float32).numpy()
            out["nucleus_overflow"][lo:hi, si] = overflow.numpy()

            q_min = torch.where(kept, q, torch.ones((), dtype=q.dtype)).min(dim=-1).values

            for di, d in enumerate(depths):
                power = 2 ** d
                out["p_active"][lo:hi, si, di] = (1.0 - (q ** power).sum(dim=-1)).numpy()
                out["p_active_slack"][lo:hi, si, di] = (
                    beyond * q_min ** (power - 1)
                ).numpy()

    return out
