"""Tournament sampling: draw 2**d candidates i.i.d. from the post-truncation
distribution and run a d-round knockout decided by key-derived g-values.

Sits after truncation, matching the reference design. Implemented as an explicit
sampler rather than a transformers LogitsProcessor because the selection reads 2**d
draws and no pure logit transform expresses that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import WatermarkConfig
from .prf import PRF, context_at
from .truncate import truncate_full

__all__ = ["TournamentSampler", "generate"]


@dataclass
class TournamentSampler:
    prf: PRF
    cfg: WatermarkConfig
    rng: np.random.Generator

    def sample(self, probs: np.ndarray, context: Sequence[int]) -> int:
        """probs must be post-truncation and renormalized over the full vocab, with
        truncated entries exactly zero."""
        n = self.cfg.n_candidates
        cdf = np.cumsum(probs)
        survivors = np.searchsorted(cdf, self.rng.random(n) * cdf[-1], side="right")
        survivors = np.clip(survivors, 0, probs.shape[0] - 1)

        if not context:
            return int(survivors[0])

        for round_idx in range(self.cfg.tournament_depth):
            left, right = survivors[0::2], survivors[1::2]
            g_left = np.fromiter(
                (self.prf.bit(context, int(c), round_idx) for c in left),
                dtype=np.uint8,
                count=left.shape[0],
            )
            g_right = np.fromiter(
                (self.prf.bit(context, int(c), round_idx) for c in right),
                dtype=np.uint8,
                count=right.shape[0],
            )
            # Uniform tiebreak, not PRF-derived: a keyed tiebreak injects bias the
            # detector never reads, changing the distortion characteristics.
            coin = self.rng.integers(0, 2, size=left.shape[0]).astype(bool)
            take_left = np.where(g_left == g_right, coin, g_left > g_right)
            survivors = np.where(take_left, left, right)

        return int(survivors[0])


def generate(
    model,
    tokenizer,
    prompt: str,
    cfg: WatermarkConfig,
    key: bytes | None,
    max_new_tokens: int,
    seed: int,
    device: str = "cpu",
) -> dict:
    """key=None runs the identical sampler with a uniform choice at every round, which
    is the unwatermarked control the Phase 0.5 gate compares against."""
    import torch

    rng = np.random.default_rng(seed)
    sampler = TournamentSampler(PRF(key), cfg, rng) if key is not None else None

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = int(ids.shape[1])
    generated: list[int] = []
    eos = tokenizer.eos_token_id

    past = None
    cur = ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float().numpy()

        probs = truncate_full(logits, cfg)
        full = np.asarray(ids[0].tolist() + generated, dtype=np.int64)
        context = context_at(full, len(full), cfg.context_depth)

        if sampler is not None:
            token = sampler.sample(probs, context)
        else:
            cdf = np.cumsum(probs)
            token = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
            token = min(token, probs.shape[0] - 1)

        generated.append(token)
        if eos is not None and token == eos:
            break
        cur = torch.tensor([[token]], device=device)

    return {
        "prompt_len": prompt_len,
        "token_ids": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "watermarked": key is not None,
    }
