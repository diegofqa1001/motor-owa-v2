"""Motor de recomendacion adaptativo v2: orquestacion del ciclo completo.

Une todas las piezas segun CRISP-DM (fase de modelado + despliegue):

    RecommendationEngine.run_cycle() ejecuta, para un inversor:
      clasificar -> recomendar (cartera sigma_k) -> invertir horizonte h ->
      cosechar -> recalibrar (posible migracion de perfil) -> repetir.

    RecommendationEngine.panel_backtest() ejecuta las 8 carteras canonicas
    en ventanas rodantes y reporta las metricas del anteproyecto (RMSE,
    NDCG@k, consistencia ordinal, coherencia orness-vol) con particion
    70-20-10.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .adaptive import InvestorState, harvest_and_recalibrate
from .config import Anchors, EngineConfig
from .criteria import CRITERIA
from .portfolio import PortfolioBuilder, PortfolioResult
from .profiles import Profile, all_profiles, get_profile
from .validation import (coherence_spearman, mae, mape, ndcg_at_k, mrr,
                         ordinal_consistency, rmse, split_70_20_10)

_ANNUAL = 252

__all__ = ["CycleRecord", "RecommendationEngine"]


@dataclass
class CycleRecord:
    """Registro auditable de un ciclo inversion-cosecha-recalibracion."""
    t: int
    profile_before: str
    profile_after: str
    portfolio: PortfolioResult
    realized_return: float
    surprise: float
    migrated: bool


class RecommendationEngine:
    """Motor adaptativo v2 sobre un panel de precios."""

    def __init__(self, prices: pd.DataFrame,
                 config: Optional[EngineConfig] = None,
                 volumes: Optional[pd.DataFrame] = None):
        self.prices = prices
        self.cfg = config or EngineConfig()
        self.builder = PortfolioBuilder(prices, self.cfg, volumes)
        self.profiles = all_profiles(self.cfg.anchors)

    # ------------------------------------------------ ciclo individual
    def realized_return(self, weights: pd.Series, t: int, h: int) -> float:
        """Retorno realizado de la cartera en (t, t+h], neto de costos."""
        t2 = min(t + h, len(self.prices) - 1)
        px = self.prices[weights.index]
        gross = float((px.iloc[t2] / px.iloc[t] - 1.0) @ weights.values)
        return gross - self.cfg.tc_bps / 1e4

    def run_cycle(self, state: InvestorState, t: int) -> CycleRecord:
        """Un ciclo completo para un inversor con estado dinamico."""
        prof_before = state.profile.name
        port = self.builder.build(state.profile, t)
        h = self.cfg.horizon
        r = self.realized_return(port.weights, t, h)
        mu_h = port.expected_return * h / _ANNUAL
        sig_h = port.expected_vol * np.sqrt(h / _ANNUAL)
        harvest_and_recalibrate(state, r, mu_h, sig_h, self.cfg)
        rec = state.history[-1]
        return CycleRecord(t=t, profile_before=prof_before,
                           profile_after=state.profile.name, portfolio=port,
                           realized_return=r, surprise=rec["surprise"],
                           migrated=rec["migrated"])

    def simulate_investor(self, state: InvestorState, t0: int,
                          n_cycles: int) -> List[CycleRecord]:
        """Trayectoria adaptativa de un inversor durante n_cycles horizontes."""
        out, t = [], t0
        for _ in range(n_cycles):
            if t + self.cfg.horizon >= len(self.prices):
                break
            out.append(self.run_cycle(state, t))
            t += self.cfg.horizon
        return out

    # ------------------------------------------------ backtest de panel
    def panel_backtest(self, step: Optional[int] = None) -> Dict[str, object]:
        """Backtest de las 8 carteras canonicas en ventanas rodantes.

        Devuelve metricas del anteproyecto sobre la particion 70-20-10:
        RMSE/MAE/MAPE (scores intermedios vs finales), NDCG@k, MRR,
        consistencia ordinal, y coherencia conductual Spearman(orness, vol).
        """
        cfg = self.cfg
        step = step or cfg.horizon
        t_grid = list(range(cfg.lookback, len(self.prices) - cfg.horizon, step))
        if len(t_grid) < 10:
            raise ValueError("Serie demasiado corta para el backtest.")
        tr, ve, va = split_70_20_10(len(t_grid))
        rows, vol_track = [], {p.name: [] for p in self.profiles}
        ret_track = {p.name: [] for p in self.profiles}
        scores_by_slice = {"train": {}, "verify": {}, "validate": {}}
        for si, sl in (("train", tr), ("verify", ve), ("validate", va)):
            for t in t_grid[sl]:
                for p in self.profiles:
                    port = self.builder.build(p, t)
                    r = self.realized_return(port.weights, t, cfg.horizon)
                    px = self.prices[port.weights.index]
                    seg = px.iloc[t:t + cfg.horizon].pct_change().dropna()
                    vol_r = float((seg @ port.weights.values).std()
                                  * np.sqrt(_ANNUAL))
                    vol_track[p.name].append(vol_r)
                    ret_track[p.name].append(r)
                    scores_by_slice[si].setdefault(p.name, []).append(
                        port.scores.reindex(self.prices.columns))
                    rows.append({"slice": si, "t": t, "profile": p.name,
                                 "alpha": p.alpha, "ret": r, "vol": vol_r,
                                 "target_vol": port.target_vol,
                                 "expected_vol": port.expected_vol})
        df = pd.DataFrame(rows)
        # --- metricas de precision entre verificacion y validacion ---
        metrics: Dict[str, object] = {}
        per_profile = {}
        for p in self.profiles:
            s_ve = pd.concat(scores_by_slice["verify"][p.name], axis=1).mean(axis=1)
            s_va = pd.concat(scores_by_slice["validate"][p.name], axis=1).mean(axis=1)
            common = s_ve.dropna().index.intersection(s_va.dropna().index)
            sv, sw = s_ve[common].values, s_va[common].values
            order_pred = np.argsort(-sv)
            k = min(cfg.top_n, len(common))
            relevant = set(np.argsort(-sw)[:k].tolist())
            per_profile[p.name] = {
                "rmse": rmse(sw, sv), "mae": mae(sw, sv), "mape": mape(sw, sv),
                "ndcg_at_k": ndcg_at_k(sw - sw.min(), order_pred, k=k),
                "mrr": mrr(list(relevant), order_pred.tolist()),
                "ordinal_consistency": ordinal_consistency(
                    pd.Series(sv).rank().values, pd.Series(sw).rank().values),
            }
        metrics["per_profile"] = per_profile
        # --- coherencia conductual (garantizada por diseno) ---
        alphas = [p.alpha for p in self.profiles]
        mean_vol = [float(np.mean(vol_track[p.name])) for p in self.profiles]
        mean_ret = [float(np.mean(ret_track[p.name])) for p in self.profiles]
        metrics["coherence_vol"] = coherence_spearman(alphas, mean_vol)
        metrics["coherence_ret"] = coherence_spearman(alphas, mean_ret)
        metrics["mean_vol"] = dict(zip([p.name for p in self.profiles], mean_vol))
        metrics["mean_ret"] = dict(zip([p.name for p in self.profiles], mean_ret))
        metrics["records"] = df
        return metrics
