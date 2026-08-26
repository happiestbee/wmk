"""The generator truncates with truncate_full and the capacity scan truncates in-pass
inside evaluate_logits. If those two disagree, the watermarked text and the measured
capacity describe different distributions and F1 is meaningless."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wmk.capacity import p_active
from wmk.config import WatermarkConfig
from wmk.grid import load_grid
from wmk.scan import evaluate_logits

GRID = load_grid()


@pytest.fixture(scope="module")
def logits() -> torch.Tensor:
    rng = np.random.default_rng(0)
    vocab = 4096
    rows = [
        rng.normal(0, 4, vocab),
        rng.normal(0, 1, vocab),
        np.full(vocab, 0.0),
    ]
    peaked = np.full(vocab, -30.0)
    peaked[7] = 20.0
    rows.append(peaked)
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def test_scan_and_generator_truncation_agree(logits):
    evaluated = evaluate_logits(logits, GRID, chunk=2)

    for si, serving in enumerate(GRID.serving_configs):
        for di, depth in enumerate(GRID.tournament_depths):
            cfg = WatermarkConfig(
                tokenizer_id="t",
                tokenizer_hash="0" * 64,
                tournament_depth=depth,
                temperature=serving.temperature,
                top_p=serving.top_p,
                top_k=serving.top_k,
            )
            reference = p_active(truncate_rows(logits, cfg), depth)
            got = evaluated["p_active"][:, si, di]
            assert got == pytest.approx(reference, abs=1e-12), (
                f"{serving.label} d={depth}: {got} != {reference}"
            )


def truncate_rows(logits: torch.Tensor, cfg: WatermarkConfig) -> np.ndarray:
    from wmk.truncate import truncate_full

    return truncate_full(logits.numpy(), cfg)


def test_truncation_renormalizes_and_zeroes_the_tail(logits):
    cfg = WatermarkConfig(
        tokenizer_id="t", tokenizer_hash="0" * 64, top_p=0.9, temperature=0.7
    )
    probs = truncate_rows(logits, cfg)
    assert np.allclose(probs.sum(axis=-1), 1.0)
    assert (probs >= 0).all()
    assert (probs == 0).any(), "top_p=0.9 must remove mass from a broad row"


def test_deterministic_row_yields_zero_capacity_under_every_config(logits):
    evaluated = evaluate_logits(logits[3:4], GRID, chunk=1)
    assert evaluated["p_active"].max() < 1e-12
