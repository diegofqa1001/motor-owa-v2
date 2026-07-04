"""Demo de un comando del motor v2.

Uso:
    python scripts/run_demo.py                # panel sintetico offline (tests)
    python scripts/run_demo.py --market us    # datos reales US (yfinance)
    python scripts/run_demo.py --market co    # datos reales Colombia (BVC)

Genera en results/ y figures/:
    - anclas de los 8 perfiles (orness ladder)
    - las 8 carteras con su volatilidad objetivo vs alcanzada (frontera)
    - trayectoria adaptativa de un inversor Pragmatist (migraciones)
    - metricas del anteproyecto (RMSE, NDCG@k, MRR, consistencia ordinal,
      coherencia orness-volatilidad) con particion 70-20-10 -> CSV
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from motor_owa.adaptive import InvestorState
from motor_owa.config import EngineConfig, TICKERS_CO, TICKERS_US
from motor_owa.data import load_yfinance, simulate_market
from motor_owa.engine import RecommendationEngine
from motor_owa.profiles import all_profiles
from motor_owa.viz import plot_adaptive_path, plot_frontier, plot_orness_ladder

HERE = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["synthetic", "us", "co"],
                    default="synthetic")
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    if args.market == "us":
        px = load_yfinance(TICKERS_US, start=args.start)
    elif args.market == "co":
        px = load_yfinance(TICKERS_CO, start=args.start)
    else:
        print("[demo] Panel SINTETICO reproducible (solo demostracion offline).")
        px = simulate_market(n_assets=20, n_days=1500)

    cfg = EngineConfig()
    eng = RecommendationEngine(px, cfg)
    profs = all_profiles(cfg.anchors)

    # 1. anclas
    plot_orness_ladder(profs, os.path.join(FIG, "fig1_orness_ladder.png"))

    # 2. las 8 carteras hoy
    t = len(px) - cfg.horizon - 1
    ports = eng.builder.build_all(profs, t)
    plot_frontier(ports, os.path.join(FIG, "fig2_frontier.png"))
    resumen = pd.DataFrame({
        n: {"orness": r.alpha, "target_vol": r.target_vol,
            "expected_vol": r.expected_vol,
            "expected_return": r.expected_return,
            "n_activos": len(r.selected),
            "activos": " ".join(r.selected)}
        for n, r in ports.items()}).T
    resumen.to_csv(os.path.join(RES, "carteras_por_perfil.csv"))
    print(resumen[["orness", "target_vol", "expected_vol",
                   "expected_return"]].round(4))

    # 3. inversor adaptativo
    st = InvestorState.from_profile("Pragmatist", cfg.anchors)
    t0 = cfg.lookback
    recs = eng.simulate_investor(st, t0, n_cycles=12)
    plot_adaptive_path(st.history, os.path.join(FIG, "fig3_adaptive_path.png"))
    mig = sum(1 for h in st.history if h["migrated"])
    print(f"\n[adaptativo] {len(recs)} ciclos, {mig} migraciones de perfil; "
          f"perfil final: {st.profile.name} (z={st.z:+.3f})")

    # 4. validacion interna (anteproyecto)
    m = eng.panel_backtest()
    per = pd.DataFrame(m["per_profile"]).T
    per.to_csv(os.path.join(RES, "metricas_por_perfil.csv"))
    pd.DataFrame({"coherence_vol": [m["coherence_vol"]],
                  "coherence_ret": [m["coherence_ret"]]}
                 ).to_csv(os.path.join(RES, "coherencia.csv"), index=False)
    m["records"].to_csv(os.path.join(RES, "backtest_registros.csv"), index=False)
    print(f"\n[validacion] coherencia orness-vol = {m['coherence_vol']:+.3f}")
    print(per.round(4))
    print(f"\nResultados en {os.path.abspath(RES)} y figuras en "
          f"{os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
