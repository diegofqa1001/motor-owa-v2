import pytest
from motor_owa.adaptive import InvestorState
from motor_owa.config import EngineConfig
from motor_owa.data import simulate_market
from motor_owa.engine import RecommendationEngine


@pytest.fixture(scope="module")
def engine():
    px = simulate_market(n_assets=20, n_days=1200, seed=7)
    return RecommendationEngine(px, EngineConfig(top_n=8, horizon=42))


def test_run_cycle(engine):
    st = InvestorState.from_profile("Analyst")
    rec = engine.run_cycle(st, 300)
    assert rec.profile_before == "Analyst"
    assert -1.0 < rec.realized_return < 2.0
    assert len(st.history) == 1


def test_simulate_investor_trajectory(engine):
    st = InvestorState.from_profile("Pragmatist")
    recs = engine.simulate_investor(st, 300, n_cycles=8)
    assert len(recs) >= 5
    assert len(st.history) == len(recs)


def test_panel_backtest_metrics(engine):
    m = engine.panel_backtest(step=63)
    assert m["coherence_vol"] > 0.7          # coherencia por diseno
    for prof, met in m["per_profile"].items():
        assert met["rmse"] >= 0
        assert 0 <= met["ndcg_at_k"] <= 1
        assert 0 <= met["ordinal_consistency"] <= 1
    assert set(m["records"]["slice"].unique()) == {"train", "verify", "validate"}
