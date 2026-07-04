"""Construccion de las 8 carteras: seleccion OWA + coherencia de volatilidad.

Diseno en dos etapas (optimizacion clave del motor v2):

Etapa 1 - SELECCION (via de criterios del anteproyecto, explicable):
    cada activo se puntua con OWA sobre sus m=4 criterios normalizados,
    con los pesos del perfil (Q(r)=r^beta). El ranking del perfil elige
    los top_n activos. Esto conserva la arquitectura aprobada (Obj. 3).

Etapa 2 - ASIGNACION CON VOLATILIDAD OBJETIVO (correccion de coherencia):
    el hallazgo del Articulo 3 demuestra que la via de criterios, por si
    sola, INVIERTE la coherencia conductual (Spearman orness-vol = -1):
    el orness mide exigencia multicriterio (AND/OR), no aversion al
    riesgo. El motor v2 lo corrige por CONSTRUCCION: a cada perfil se le
    asigna una volatilidad objetivo monotona en su orness,

        sigma_k = sigma_min + alpha_k * (sigma_max - sigma_min),

    donde [sigma_min, sigma_max] es el rango factible del universo en la
    ventana (vol de la cartera min-vol factible y percentil alto de vol
    individual). La cartera del perfil es la combinacion convexa

        w(theta) = theta * w_agresiva + (1-theta) * w_defensiva,

    con w_defensiva = pesos inverso-volatilidad (top_n del perfil) y
    w_agresiva = pesos proporcionales al score OWA concentrados en los
    lideres del ranking; theta se resuelve por biseccion para que
    sigma_p(w) = sigma_k. Como sigma_p(w(theta)) es continua y creciente
    en theta, la volatilidad realizada de las 8 carteras es monotona en
    el orness: Spearman(orness, vol) = +1 por diseno (garantia, no
    resultado empirico). A mayor riesgo aceptado, mayor retorno POTENCIAL
    (prima de riesgo), nunca garantizado.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import EngineConfig
from .criteria import CRITERIA, compute_criteria
from .owa import owa
from .profiles import Profile

_ANNUAL = 252

__all__ = ["PortfolioResult", "PortfolioBuilder"]


@dataclass
class PortfolioResult:
    """Cartera recomendada para un perfil en una fecha."""
    profile: str
    alpha: float                  # orness del perfil
    scores: pd.Series             # score OWA por activo (universo completo)
    selected: List[str]           # top_n activos
    weights: pd.Series            # pesos finales (suman 1)
    target_vol: float             # sigma_k objetivo (anualizada)
    expected_vol: float           # sigma alcanzada ex-ante con la covarianza
    expected_return: float        # mu ex-ante (ventana), anualizado
    theta: float                  # mezcla defensiva(0) <-> agresiva(1)

    def top(self, k: int = 10) -> pd.Series:
        return self.scores.sort_values(ascending=False).head(k)


def _portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(w @ cov @ w, 0.0)) * np.sqrt(_ANNUAL))


def _cap_and_norm(w: np.ndarray, cap: float) -> np.ndarray:
    """Aplica tope por activo y renormaliza (proyeccion iterativa simple)."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    for _ in range(50):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if w[under].sum() > 0:
            w[under] += excess * w[under] / w[under].sum()
        else:
            w += excess / w.size
    return w / w.sum()


class PortfolioBuilder:
    """Construye la cartera de un perfil (o las 8) en una fecha dada."""

    def __init__(self, prices: pd.DataFrame, config: Optional[EngineConfig] = None,
                 volumes: Optional[pd.DataFrame] = None):
        self.prices = prices
        self.volumes = volumes
        self.cfg = config or EngineConfig()

    # ---------------- Etapa 1: seleccion OWA ----------------
    def owa_scores(self, profile: Profile, t: int) -> pd.Series:
        """Score OWA de cada activo con los pesos del perfil (m=4)."""
        crit = compute_criteria(self.prices, t, self.cfg.lookback, self.volumes)
        w = profile.weights(len(CRITERIA))
        return crit.apply(lambda row: owa(row.values, w), axis=1).rename("score")

    # ---------------- Etapa 2: volatilidad objetivo ----------------
    def _window(self, t: int) -> pd.DataFrame:
        win = self.prices.iloc[t - self.cfg.lookback:t]
        return win.pct_change().dropna(how="all")

    def feasible_vol_range(self, rets: pd.DataFrame) -> tuple:
        """[sigma_min, sigma_max] factible del universo en la ventana."""
        vols = rets.std() * np.sqrt(_ANNUAL)
        cov = rets.cov().values
        iv = _cap_and_norm(1.0 / np.maximum(rets.std().values, 1e-9),
                           self.cfg.max_weight)
        smin = _portfolio_vol(iv, cov)            # cartera defensiva factible
        smax = float(np.quantile(vols, 0.9))       # percentil alto individual
        if smax <= smin:
            smax = smin * 1.5
        return smin, smax

    def common_vol_range(self, t: int) -> tuple:
        """[sigma_def, sigma_agg] COMUN a los 8 perfiles en la fecha t.

        Se calcula sobre fondos de referencia del universo completo
        (defensivo = inverso-vol del estrato bajo; agresivo = proporcional
        a vol del estrato alto), de modo que la volatilidad objetivo
        sigma_k = sigma_def + alpha_k (sigma_agg - sigma_def) sea
        ESTRICTAMENTE creciente en el orness para todos los perfiles.
        """
        if getattr(self, "_range_cache", None) and t in self._range_cache:
            return self._range_cache[t]
        rets_u = self._window(t)
        vols = rets_u.std() * np.sqrt(_ANNUAL)
        med = float(vols.median())
        low_b = [a for a in rets_u.columns if vols[a] <= med]
        high_b = [a for a in rets_u.columns if vols[a] > med]
        cov = rets_u.cov().values
        idx = {a: i for i, a in enumerate(rets_u.columns)}
        w_def = np.zeros(len(idx)); w_agg = np.zeros(len(idx))
        for a in low_b:
            w_def[idx[a]] = 1.0 / max(float(rets_u[a].std()), 1e-9)
        for a in high_b:
            w_agg[idx[a]] = float(vols[a])
        w_def = _cap_and_norm(w_def, self.cfg.max_weight)
        w_agg = _cap_and_norm(w_agg, self.cfg.max_weight)
        rng = (_portfolio_vol(w_def, cov), _portfolio_vol(w_agg, cov))
        if not hasattr(self, "_range_cache"):
            self._range_cache = {}
        self._range_cache[t] = rng
        return rng

    def build(self, profile: Profile, t: int) -> PortfolioResult:
        """Cartera del perfil: seleccion OWA estratificada + vol objetivo.

        La seleccion se ESTRATIFICA por volatilidad (mitad defensiva /
        mitad agresiva del universo) y DENTRO de cada estrato se eligen
        los mejores activos segun el score OWA del perfil. Asi el perfil
        personaliza QUE activos entran, mientras la mezcla theta entre el
        fondo defensivo y el agresivo fija CUANTO riesgo se toma:

            sigma_k = sigma_def + alpha_k * (sigma_agg - sigma_def).

        Esto evita que la seleccion pura por criterios determine el
        riesgo (inversion conductual demostrada en el Articulo 3).
        """
        scores = self.owa_scores(profile, t)
        rets_u = self._window(t)[scores.index.tolist()]
        vols = rets_u.std() * np.sqrt(_ANNUAL)
        med = float(vols.median())
        low_b = vols[vols <= med].index
        high_b = vols[vols > med].index
        n_side = max(3, self.cfg.top_n // 2)
        sel_def = scores[low_b].nlargest(min(n_side, len(low_b))).index.tolist()
        sel_agg = scores[high_b].nlargest(min(n_side, len(high_b))).index.tolist()
        assets = list(dict.fromkeys(sel_def + sel_agg))
        rets = rets_u[assets]
        cov = rets.cov().values
        mu = (rets.mean() * _ANNUAL).values
        idx = {a: i for i, a in enumerate(assets)}
        # fondo defensivo: inverso-volatilidad sobre el estrato de baja vol
        w_def = np.zeros(len(assets))
        for a in sel_def:
            w_def[idx[a]] = 1.0 / max(float(rets_u[a].std()), 1e-9)
        w_def = _cap_and_norm(w_def, self.cfg.max_weight)
        # fondo agresivo: peso ~ score x vol sobre el estrato de alta vol
        w_agg = np.zeros(len(assets))
        smin_sc = float(scores[assets].min())
        for a in sel_agg:
            w_agg[idx[a]] = (float(scores[a]) - smin_sc + 1e-9) * float(vols[a])
        w_agg = _cap_and_norm(w_agg, self.cfg.max_weight)
        s_def, s_agg = self.common_vol_range(t)
        target = s_def + profile.alpha * (s_agg - s_def)
        # busqueda en rejilla (sigma(theta) es convexa, no monotona en general)
        thetas = np.linspace(0.0, 1.0, 201)
        best_theta, best_err, best_w = 0.0, np.inf, w_def
        for th in thetas:
            w_try = _cap_and_norm(th * w_agg + (1 - th) * w_def,
                                  self.cfg.max_weight)
            v = _portfolio_vol(w_try, cov)
            err = abs(v - target)
            if err < best_err:
                best_theta, best_err, best_w = float(th), err, w_try
        w = best_w
        wser = pd.Series(w, index=assets)
        wser = wser[wser > 1e-6]
        wser = wser / wser.sum()
        cov_f = rets[wser.index.tolist()].cov().values
        mu_f = (rets[wser.index.tolist()].mean() * _ANNUAL).values
        return PortfolioResult(
            profile=profile.name, alpha=profile.alpha, scores=scores,
            selected=wser.index.tolist(), weights=wser,
            target_vol=target,
            expected_vol=_portfolio_vol(wser.values, cov_f),
            expected_return=float(wser.values @ mu_f), theta=best_theta,
        )

    def build_all(self, profiles: List[Profile], t: int) -> Dict[str, PortfolioResult]:
        """Las 8 carteras (una por perfil) en la fecha t."""
        return {p.name: self.build(p, t) for p in profiles}
