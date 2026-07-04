import numpy as np
import pytest
from motor_owa.adaptive import (InvestorState, surprise, update_latent,
                                harvest_and_recalibrate)
from motor_owa.config import EngineConfig
from motor_owa.latent import octile_z


def test_surprise_sign():
    assert surprise(0.10, 0.05, 0.10) > 0
    assert surprise(-0.10, 0.05, 0.10) < 0


def test_update_latent_direction():
    z = 0.0
    assert update_latent(z, +1.0) > z
    assert update_latent(z, -1.0) < z


def test_loss_aversion_asymmetry():
    z = 0.0
    up = update_latent(z, +1.0) - z
    down = z - update_latent(z, -1.0)
    assert down > up            # las perdidas pesan mas (lambda=2.25)
    assert down / up == pytest.approx(2.25, rel=1e-6)


def test_update_bounded():
    assert update_latent(3.0, 100.0) <= 3.0
    assert update_latent(-3.0, -100.0) >= -3.0


def test_state_from_profile_and_migration():
    st = InvestorState.from_profile("Pragmatist")
    assert st.k == 3
    cfg = EngineConfig(kappa=0.6)
    # racha de perdidas inesperadas -> migra hacia conservador
    for _ in range(5):
        harvest_and_recalibrate(st, realized=-0.20, expected=0.03,
                                expected_vol=0.05, cfg=cfg)
    assert st.k < 3
    assert any(h["migrated"] for h in st.history)


def test_good_harvest_moves_up():
    st = InvestorState.from_profile("Pragmatist")
    cfg = EngineConfig(kappa=0.6)
    for _ in range(6):
        harvest_and_recalibrate(st, realized=0.30, expected=0.03,
                                expected_vol=0.05, cfg=cfg)
    assert st.k > 3


def test_history_traceability():
    st = InvestorState.from_profile(1)
    harvest_and_recalibrate(st, 0.02, 0.01, 0.05)
    h = st.history[-1]
    for key in ("surprise", "z_before", "z_after", "k_before", "k_after",
                "migrated"):
        assert key in h
