"""verify_vol_range_real_data.py -- Hallazgo 3 (revision postdoctoral 2026-08-17):
verifica empiricamente, sobre datos reales de mercado, cuantas veces se activa
la cota de seguridad de common_vol_range() (ver PortfolioBuilder.vol_range_violations
en src/motor_owa/portfolio.py).

Contexto: la garantia de coherencia conductual (Spearman orness-vol = +1 por
diseno) esta CONDICIONADA a que sigma_agg > sigma_def en cada ventana. Ese
supuesto es empiricamente muy probable pero no esta garantizado por
construccion. Este script re-ejecuta el panel_backtest() completo del motor
2.1 sobre datos reales de EE. UU. y Colombia y reporta cuantas de esas
ventanas violaron el supuesto (y por tanto activaron la cota de seguridad
sigma_agg = 1.5 * sigma_def), para que la frecuencia real de la excepcion
quede documentada y no asumida.

Uso:  python scripts/verify_vol_range_real_data.py
Requiere red (descarga precios via motor_owa.data.load_yfinance).
"""
from __future__ import annotations

import sys

from motor_owa.config import EngineConfig, TICKERS_CO, TICKERS_US
from motor_owa.data import load_yfinance
from motor_owa.engine import RecommendationEngine


def check_market(nombre: str, tickers: list[str]) -> None:
    print(f"[verify] descargando {nombre} ({len(tickers)} activos)...")
    px = load_yfinance(tickers, start="2015-01-01")
    print(f"[verify] {nombre}: {px.shape[0]} dias x {px.shape[1]} activos "
          f"({px.index.min().date()} .. {px.index.max().date()})")

    cfg = EngineConfig()
    eng = RecommendationEngine(px, cfg)
    eng.panel_backtest()
    b = eng.builder

    tasa = (100 * b.vol_range_violations / b.vol_range_calls
            if b.vol_range_calls else float("nan"))
    print(f"[verify] {nombre}: vol_range_calls={b.vol_range_calls}  "
          f"vol_range_violations={b.vol_range_violations}  "
          f"tasa={tasa:.3f}%\n")


def main() -> None:
    check_market("EE. UU.", TICKERS_US)
    check_market("Colombia", TICKERS_CO)


if __name__ == "__main__":
    sys.exit(main())
