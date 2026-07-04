import numpy as np
import pytest
from motor_owa.quantifiers import (rim_weights, exponent_for_orness,
                                   weights_for_orness, orness_limit)
from motor_owa.owa import orness, owa


def test_rim_weights_sum_one():
    for n in (2, 4, 7, 20):
        for b in (0.2, 1.0, 3.7):
            w = rim_weights(n, b)
            assert w.sum() == pytest.approx(1.0)
            assert (w >= 0).all()


def test_beta_one_uniform():
    w = rim_weights(5, 1.0)
    assert np.allclose(w, 0.2)
    assert orness(w) == pytest.approx(0.5)


def test_exponent_recovers_orness_exactly():
    for n in (4, 7):
        for target in [1/16, 3/16, 0.158, 0.5, 0.693, 0.865, 15/16]:
            b = exponent_for_orness(n, target)
            assert orness(rim_weights(n, b)) == pytest.approx(target, abs=1e-8)


def test_same_orness_different_n_different_beta():
    b4 = exponent_for_orness(4, 0.0625)
    b7 = exponent_for_orness(7, 0.0625)
    assert b4 != pytest.approx(b7, abs=1e-3)


def test_orness_limit_continuous():
    assert orness_limit(1.0) == pytest.approx(0.5)
    # con n grande el orness discreto converge al limite 1/(beta+1)
    assert orness(rim_weights(5000, 3.0)) == pytest.approx(0.25, abs=1e-3)


def test_owa_min_max_behavior():
    vals = [0.9, 0.2, 0.6]
    w_or = weights_for_orness(3, 0.999)   # ~max
    w_and = weights_for_orness(3, 0.001)  # ~min
    assert owa(vals, w_or) == pytest.approx(0.9, abs=0.01)
    assert owa(vals, w_and) == pytest.approx(0.2, abs=0.01)
