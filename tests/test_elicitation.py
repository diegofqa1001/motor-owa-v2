import numpy as np
import pytest
from motor_owa.adaptive import InvestorState, harvest_and_recalibrate
from motor_owa.config import EngineConfig, DIMENSIONS
from motor_owa.elicitation import (QUESTIONNAIRE, declared_z, scores_from_z,
                                   simulate_declared_scores,
                                   emotional_gap_metrics)
from motor_owa.latent import classify_z


def test_questionnaire_covers_all_dimensions():
    keys = [q[0] for q in QUESTIONNAIRE]
    assert set(keys) == set(DIMENSIONS)
    assert len(keys) == 7


def test_declared_z_neutral_and_extremes():
    neutral = {k: 3.0 for k in DIMENSIONS}
    assert declared_z(neutral) == pytest.approx(0.0, abs=1e-9)
    assert classify_z(declared_z({k: 5.0 for k in DIMENSIONS})) == 8
    assert classify_z(declared_z({k: 1.0 for k in DIMENSIONS})) == 1


def test_scores_from_z_roundtrip():
    for z in (-1.5, -0.5, 0.0, 0.5, 1.5):
        sc = scores_from_z(z, noise_sd=0.0)
        assert declared_z(sc) == pytest.approx(z, abs=0.15)


def test_declared_channel_overrides_model():
    st = InvestorState.from_profile("Pragmatist")
    aggressive = {k: 5.0 for k in DIMENSIONS}
    harvest_and_recalibrate(st, realized=-0.10, expected=0.02,
                            expected_vol=0.05,
                            declared_scores=aggressive)
    # el modelo predice bajar; el decisor declara subir: manda el decisor
    h = st.history[-1]
    assert h["z_model"] < h["z_before"]
    assert st.z > h["z_before"]
    assert h["epsilon"] is not None and h["epsilon"] > 0
    assert h["declared"] is True


def test_wealth_capitalizes():
    st = InvestorState.from_profile(4)
    st.wealth = 100.0
    harvest_and_recalibrate(st, realized=0.10, expected=0.02, expected_vol=0.05)
    assert st.wealth == pytest.approx(110.0)


def test_logical_decider_has_near_zero_gap():
    """Control negativo: decisor sin emocion -> brecha ~0."""
    rng = np.random.default_rng(3)
    cfg = EngineConfig()
    st = InvestorState.from_profile("Analyst")
    for i in range(30):
        st.z = 0.0     # un paso por ciclo: aisla el mecanismo de la
                       # saturacion del instrumento (escala acotada)
        s = float(rng.normal(0, 1.2))
        dec = simulate_declared_scores(st.z, s, cfg.kappa, cfg.loss_lambda,
                                       emotion_gain=0.0, noise_sd=0.0, rng=rng)
        harvest_and_recalibrate(st, realized=0.02 + 0.05 * s, expected=0.02,
                                expected_vol=0.05, cfg=cfg,
                                declared_scores=dec)
    m = emotional_gap_metrics(st.history)
    assert m["n"] == 30
    assert abs(m["mean_gap"]) < 0.05
    assert m["mean_abs_gap"] < 0.10


def test_emotional_decider_is_detected():
    """Control positivo: emocion sembrada -> brecha detectable y correlada."""
    rng = np.random.default_rng(7)
    cfg = EngineConfig()
    st = InvestorState.from_profile("Analyst")
    for i in range(40):
        st.z = 0.0
        s = float(rng.normal(0, 1.2))
        sentiment = np.sign(s)          # euforia/panico alineado a la sorpresa
        dec = simulate_declared_scores(st.z, s, cfg.kappa, cfg.loss_lambda,
                                       emotion_gain=0.5, sentiment=sentiment,
                                       noise_sd=0.05, rng=rng)
        harvest_and_recalibrate(st, realized=0.02 + 0.05 * s, expected=0.02,
                                expected_vol=0.05, cfg=cfg,
                                declared_scores=dec)
    m = emotional_gap_metrics(st.history)
    assert m["mean_abs_gap"] > 0.15          # hay brecha emocional
    assert m["corr_gap_surprise"] > 0.4      # correlada con la sorpresa


def test_lambda_recovered_from_declarations():
    """La asimetria de perdida se ESTIMA de las declaraciones (no se asume)."""
    rng = np.random.default_rng(11)
    cfg = EngineConfig()
    st = InvestorState.from_profile("Strategist")
    for i in range(60):
        st.z = 0.0
        s = float(rng.normal(0, 1.0))
        dec = simulate_declared_scores(st.z, s, cfg.kappa, cfg.loss_lambda,
                                       emotion_gain=0.0, noise_sd=0.0, rng=rng)
        harvest_and_recalibrate(st, realized=0.02 + 0.05 * s, expected=0.02,
                                expected_vol=0.05, cfg=cfg,
                                declared_scores=dec)
    m = emotional_gap_metrics(st.history)
    assert m["lambda_hat"] == pytest.approx(2.25, rel=0.35)
