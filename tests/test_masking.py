"""Repeated-context masking is the load-bearing correction in this project: N_eff and
the detector must agree on the unmasked set, or every F1 number comes out optimistic."""

from __future__ import annotations

import numpy as np
import pytest

from wmk.capacity import document_n_eff, n_eff
from wmk.config import WatermarkConfig
from wmk.keys import null_key
from wmk.masking import masked_fraction, scorable_mask
from wmk.score import score_token_ids

CFG = WatermarkConfig(tokenizer_id="test", tokenizer_hash="0" * 64)


def _repeated(sentence: list[int], times: int) -> list[int]:
    return sentence * times


def test_repeated_sentence_scores_at_the_null_median():
    """Catches window-size errors, ring-buffer off-by-one and context-hash collisions
    in one assertion: a document that is one sentence 500 times carries no more signal
    than the sentence itself."""
    sentence = [101, 202, 303, 404, 505, 606, 707, 808]
    doc = _repeated(sentence, 500)

    res = score_token_ids(doc, CFG, null_key(0))
    assert res.n_scored <= len(sentence) + CFG.context_depth
    assert res.masked_fraction > 0.99
    assert abs(res.mean_g - 0.5) < 0.5

    unmasked = score_token_ids(doc, CFG.with_(masking_window=1), null_key(0))
    assert unmasked.n_scored > 100 * res.n_scored


def test_masking_window_bounds_the_lookback():
    period = 16
    doc = _repeated(list(range(period)), 80)

    narrow = scorable_mask(doc, CFG.context_depth, masking_window=period // 2)
    wide = scorable_mask(doc, CFG.context_depth, masking_window=len(doc))
    assert narrow.sum() > wide.sum()

    novel = list(range(1000, 1000 + len(doc)))
    assert scorable_mask(novel, CFG.context_depth, 512).sum() == len(novel) - CFG.context_depth


def test_short_and_degenerate_documents_do_not_raise():
    for doc in ([], [7], [7, 7], [7] * (CFG.context_depth + 1)):
        mask = scorable_mask(doc, CFG.context_depth, CFG.masking_window)
        assert mask.shape[0] == len(doc)
        assert 0.0 <= masked_fraction(mask, CFG.context_depth) <= 1.0


def test_capacity_and_detector_agree_on_the_unmasked_set():
    """Phase 3 sums p_active over unmasked positions and Phase 5 accumulates g-values
    over them. They share scorable_mask; this asserts they still do."""
    rng = np.random.default_rng(0)
    doc = rng.integers(0, 5000, size=900).tolist()
    doc[400:500] = doc[300:400]

    probs = np.full((len(doc), 8), 1.0 / 8)
    detector = score_token_ids(doc, CFG, null_key(3), p_active_values=None)
    capacity, pa, mask = document_n_eff(
        probs, np.asarray(doc), CFG.context_depth, CFG.masking_window, CFG.tournament_depth
    )

    assert detector.n_scored == int(mask.sum())
    assert capacity == pytest.approx(n_eff(pa, mask))

    with_pa = score_token_ids(doc, CFG, null_key(3), p_active_values=pa)
    assert with_pa.n_eff == pytest.approx(capacity)
