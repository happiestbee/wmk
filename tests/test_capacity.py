from fractions import Fraction

import numpy as np
import pytest

from wmk.capacity import n_eff, n_eff_min, p_active

DEPTHS = [1, 2, 4, 8]


@pytest.mark.parametrize("d", DEPTHS)
def test_deterministic_position_carries_no_signal(d):
    one_hot = np.zeros((3, 16))
    one_hot[:, 5] = 1.0
    assert np.allclose(p_active(one_hot, d), 0.0, atol=0.0)


@pytest.mark.parametrize("d", DEPTHS)
@pytest.mark.parametrize("k", [2, 7, 64])
def test_uniform_support_matches_closed_form(d, k):
    probs = np.full((1, k), 1.0 / k)
    expected = 1.0 - k * (1.0 / k) ** (2 ** d)
    assert p_active(probs, d)[0] == pytest.approx(expected, rel=1e-12, abs=1e-300)


@pytest.mark.parametrize("d", DEPTHS)
def test_near_deterministic_position_survives_cancellation(d):
    """A float32 accumulator leaves percent-level error here while passing every
    other identity in this file."""
    v, eps = 64, Fraction(1, 10 ** 6)
    exact_probs = [1 - eps] + [eps / (v - 1)] * (v - 1)
    expected = float(1 - sum(p ** (2 ** d) for p in exact_probs))

    probs = np.array([[float(p) for p in exact_probs]])
    assert p_active(probs, d)[0] == pytest.approx(expected, rel=1e-9)


def test_n_eff_ignores_masked_positions():
    pa = np.array([0.5, 0.9, 0.2, 0.4])
    assert n_eff(pa, np.array([True, False, True, False])) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        n_eff(pa, np.array([True, False]))


def test_n_eff_min_scales_as_inverse_delta_squared():
    a = n_eff_min(delta=0.02, d=2, alpha=1e-3)
    b = n_eff_min(delta=0.04, d=2, alpha=1e-3)
    assert a / b == pytest.approx(4.0, rel=1e-9)
    assert n_eff_min(delta=0.02, d=4, alpha=1e-3) == pytest.approx(a / 2.0, rel=1e-9)
