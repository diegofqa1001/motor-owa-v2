import numpy as np
import pytest
from motor_owa.config import EngineConfig
from motor_owa.data import simulate_market
from motor_owa.portfolio import PortfolioBuilder
from motor_owa.profiles import all_profiles
from motor_owa.validation import coherence_spearman


@pytest.fixture(scope="module")
def market():
    return simulate_market(n_assets=20, n_days=800, seed=42)


@pytest.fixture(scope="module")
def builder(market):
    return PortfolioBuilder(market, EngineConfig(top_n=8))


def test_weights_valid(builder):
    for p in all_profiles():
        r = builder.build(p, 300)
        assert r.weights.sum() == pytest.approx(1.0)
        assert (r.weights >= -1e-12).all()
        cap_eff = max(builder.cfg.max_weight, 1.0 / len(r.selected))
        assert (r.weights <= cap_eff + 1e-6).all()
        assert 3 <= len(r.selected) <= 8


def test_target_vol_monotone_in_orness(builder):
    profs = all_profiles()
    res = [builder.build(p, 300) for p in profs]
    targets = [r.target_vol for r in res]
    assert targets == sorted(targets)


def test_expected_vol_coherent(builder):
    profs = all_profiles()
    res = [builder.build(p, 300) for p in profs]
    rho = coherence_spearman([p.alpha for p in profs],
                             [r.expected_vol for r in res])
    assert rho > 0.9  # coherencia por construccion


def test_conservative_less_volatile_than_aggressive(builder):
    profs = all_profiles()
    r1 = builder.build(profs[0], 300)   # Guardian
    r8 = builder.build(profs[7], 300)   # Visionary
    assert r1.expected_vol < r8.expected_vol
