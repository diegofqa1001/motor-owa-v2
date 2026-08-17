"""Pruebas de la salvaguarda de common_vol_range() (revision postdoctoral
2026-08-17): la monotonia de sigma_k = sigma_def + alpha_k*(sigma_agg-sigma_def)
en el orness requiere sigma_agg > sigma_def. Estas pruebas verifican (a) que
la cota de seguridad activa correctamente cuando esa premisa se viola, y
(b) que en un mercado tipico (bien diversificado, sin patologias de
covarianza) la cota NO se activa, es decir, que la correccion no altera el
comportamiento normal documentado en test_portfolio.py.
"""
import numpy as np
import pytest
from motor_owa.config import EngineConfig
from motor_owa.data import simulate_market
from motor_owa import portfolio as portfolio_mod
from motor_owa.portfolio import PortfolioBuilder


def test_safety_clamp_activates_when_premise_violated(monkeypatch):
    """Si el fondo 'agresivo' resultara con MENOR volatilidad de cartera
    que el 'defensivo' (posible bajo estructuras de covarianza adversas,
    p. ej. un estrato defensivo concentrado y correlacionado frente a un
    estrato agresivo bien diversificado), common_vol_range() debe corregir
    con la misma cota que ya usa feasible_vol_range(), no propagar la
    inversion silenciosamente."""
    market = simulate_market(n_assets=12, n_days=400, seed=1)
    cfg = EngineConfig(lookback=126, max_weight=0.30)
    b = PortfolioBuilder(market, cfg)

    # Fuerza el caso adverso: el fondo agresivo (segunda llamada a
    # _portfolio_vol dentro de common_vol_range) resulta MENOS volatil
    # que el defensivo (primera llamada).
    calls = {"n": 0}
    real_portfolio_vol = portfolio_mod._portfolio_vol

    def fake_portfolio_vol(w, cov):
        calls["n"] += 1
        return 0.12 if calls["n"] == 1 else 0.09  # s_def=0.12, s_agg=0.09 (violacion)

    monkeypatch.setattr(portfolio_mod, "_portfolio_vol", fake_portfolio_vol)
    s_def, s_agg = b.common_vol_range(300)
    monkeypatch.setattr(portfolio_mod, "_portfolio_vol", real_portfolio_vol)

    assert s_def == pytest.approx(0.12)
    assert s_agg > s_def, "la cota de seguridad debe restaurar sigma_agg > sigma_def"
    assert s_agg == pytest.approx(0.12 * 1.5)  # misma cota que feasible_vol_range()
    assert b.vol_range_violations == 1
    assert b.vol_range_calls == 1


def test_no_false_positives_in_well_behaved_market():
    """En un mercado simulado tipico (sin patologias de covarianza
    deliberadamente inducidas), la cota de seguridad no debe activarse:
    la correccion es para el caso adverso, no un cambio de comportamiento
    en el caso normal ya cubierto por test_portfolio.py."""
    market = simulate_market(n_assets=20, n_days=800, seed=42)
    cfg = EngineConfig(lookback=126, max_weight=0.30)
    b = PortfolioBuilder(market, cfg)
    for t in range(200, 750, 50):
        b.common_vol_range(t)
    assert b.vol_range_calls > 0
    assert b.vol_range_violations == 0, (
        "no se esperaban violaciones en un mercado simulado bien diversificado; "
        "si aparecen, documentar la frecuencia real en el capitulo de "
        "validacion en vez de asumir monotonia incondicional."
    )
