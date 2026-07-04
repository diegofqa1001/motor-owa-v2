"""Criterios financieros por activo (anteproyecto, Obj. 3: m = 4).

c1 Rentabilidad esperada  -- retorno medio anualizado de la ventana.
c2 Baja volatilidad       -- volatilidad anualizada, codificada inversa.
c3 Baja caida maxima      -- maximum drawdown, codificado inverso.
c4 Liquidez/estabilidad   -- proxy: volumen medio normalizado si existe;
                             si no, estabilidad (1 - autovol de la vol).

Todos se normalizan min-max al intervalo [0,1] en seccion cruzada
(anteproyecto: "normalizacion de los criterios en [0,1] para garantizar
compatibilidad agregativa"), de modo que 1 = mejor en TODOS los criterios.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["CRITERIA", "compute_criteria", "minmax_01"]

CRITERIA = ["return", "low_vol", "low_drawdown", "liquidity"]

_ANNUAL = 252


def minmax_01(s: pd.Series) -> pd.Series:
    """Normalizacion min-max a [0,1]; serie constante -> 0.5."""
    rng = s.max() - s.min()
    if rng <= 0 or not np.isfinite(rng):
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / rng


def _max_drawdown(prices: pd.Series) -> float:
    """Maxima caida (en valor absoluto) de la trayectoria de precios."""
    cummax = prices.cummax()
    dd = prices / cummax - 1.0
    return float(-dd.min())


def compute_criteria(prices: pd.DataFrame, t: int, lookback: int = 126,
                     volumes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Matriz de decision [activos x 4 criterios] en la fecha-indice t.

    prices : DataFrame de precios ajustados (filas=dias, columnas=activos).
    t      : indice entero de la fecha de evaluacion (usa (t-lookback, t]).
    """
    if t < lookback:
        raise ValueError("t debe ser >= lookback.")
    win = prices.iloc[t - lookback:t]
    rets = win.pct_change().dropna(how="all")
    mu = rets.mean() * _ANNUAL                       # rentabilidad esperada
    vol = rets.std() * np.sqrt(_ANNUAL)              # volatilidad anualizada
    mdd = win.apply(_max_drawdown)                   # caida maxima
    if volumes is not None:
        liq = volumes.iloc[t - lookback:t].mean()
    else:
        # estabilidad: volatilidad de la volatilidad rodante (inversa)
        roll = rets.rolling(21).std()
        liq = -roll.std()
    out = pd.DataFrame({
        "return": minmax_01(mu),
        "low_vol": minmax_01(-vol),
        "low_drawdown": minmax_01(-mdd),
        "liquidity": minmax_01(liq),
    })
    return out.dropna()
