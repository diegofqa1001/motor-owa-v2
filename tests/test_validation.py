import numpy as np
import pytest
from motor_owa.validation import (rmse, mae, mape, ndcg_at_k, mrr,
                                  ordinal_consistency, split_70_20_10,
                                  coherence_spearman, spearman)


def test_error_metrics_zero_when_equal():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0 and mae(y, y) == 0 and mape(y, y) == 0


def test_rmse_known_value():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_ndcg_perfect_and_worst():
    rel = [3.0, 2.0, 1.0, 0.0]
    assert ndcg_at_k(rel, [0, 1, 2, 3], k=4) == pytest.approx(1.0)
    assert ndcg_at_k(rel, [3, 2, 1, 0], k=4) < 1.0


def test_mrr():
    assert mrr([5], [1, 5, 3]) == pytest.approx(0.5)
    assert mrr([9], [1, 2, 3]) == 0.0


def test_ordinal_consistency():
    assert ordinal_consistency([1, 2, 3], [1, 2, 3]) == 1.0
    assert ordinal_consistency([1, 2, 3], [3, 2, 1]) == 0.0


def test_split_70_20_10():
    tr, ve, va = split_70_20_10(100)
    assert (tr.stop, ve.stop, va.stop) == (70, 90, 100)


def test_spearman_signs():
    a = [1, 2, 3, 4, 5]
    assert spearman(a, a) == pytest.approx(1.0)
    assert spearman(a, a[::-1]) == pytest.approx(-1.0)
    assert coherence_spearman(a, a) == pytest.approx(1.0)
