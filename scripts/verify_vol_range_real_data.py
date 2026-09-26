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

Uso:  python scripts/verify_vol_range_real_data.py [--snapshot DIR]
Por defecto lee el snapshot versionado del OE4
(../validacion-oe4/data/snapshot_oe4/{us,co}_precios.csv, con SHA-256 en su
MANIFEST.json) y escribe el registro en results/verificacion_vol_range.csv.
Con --snapshot yahoo descarga en vivo (no citable).
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from motor_owa.config import EngineConfig, TICKERS_CO, TICKERS_US
from motor_owa.data import load_csv, load_yfinance
from motor_owa.engine import RecommendationEngine


def check_market(nombre: str, tickers: list[str], snapshot: str,
                 mkt: str) -> dict:
    if snapshot == "yahoo":
        print(f"[verify] descargando {nombre} ({len(tickers)} activos)...")
        px = load_yfinance(tickers, start="2015-01-01")
    else:
        px = load_csv(os.path.join(snapshot, f"{mkt}_precios.csv"))
    print(f"[verify] {nombre}: {px.shape[0]} dias x {px.shape[1]} activos "
          f"({px.index.min().date()} .. {px.index.max().date()})")

    cfg = EngineConfig()
    eng = RecommendationEngine(px, cfg)
    m = eng.panel_backtest()
    b = eng.builder
    n_vent = int(m["records"]["t"].nunique())

    tasa = (100 * b.vol_range_violations / b.vol_range_calls
            if b.vol_range_calls else float("nan"))
    print(f"[verify] {nombre}: vol_range_calls={b.vol_range_calls}  "
          f"vol_range_violations={b.vol_range_violations}  "
          f"tasa={tasa:.3f}%  ventanas={n_vent}\n")
    return {"mercado": mkt, "jornadas": len(px), "activos": px.shape[1],
            "primera_fecha": str(px.index.min().date()),
            "ultima_fecha": str(px.index.max().date()),
            "ventanas": n_vent, "vol_range_calls": b.vol_range_calls,
            "vol_range_violations": b.vol_range_violations,
            "coherence_vol": m["coherence_vol"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.join(
        HERE, "..", "..", "validacion-oe4", "data", "snapshot_oe4"))
    args = ap.parse_args()
    rows = [check_market("EE. UU.", TICKERS_US, args.snapshot, "us"),
            check_market("Colombia", TICKERS_CO, args.snapshot, "co")]
    if args.snapshot != "yahoo":
        out = os.path.join(HERE, "..", "results", "verificacion_vol_range.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print("registro:", out)


if __name__ == "__main__":
    sys.exit(main())
