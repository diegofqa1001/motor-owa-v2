"""Datos de mercado: universos CO (BVC) y US via yfinance, con fallback CSV.

Default del motor v2 (decision 2026-07-04): datos REALES CO + US, como en el
Articulo 3. Si no hay red, ``load_csv`` permite reproducir con archivos
locales (data/*.csv con columna Date + una columna por ticker), y
``simulate_market`` genera un panel GBM multivariado SOLO para pruebas
unitarias y demos offline (etiquetado como sintetico, nunca para resultados).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .config import TICKERS_CO, TICKERS_US

__all__ = ["load_yfinance", "load_csv", "simulate_market",
           "TICKERS_CO", "TICKERS_US"]


def load_yfinance(tickers: List[str], start: str = "2015-01-01",
                  end: Optional[str] = None) -> pd.DataFrame:
    """Precios ajustados desde Yahoo Finance (requiere red y yfinance)."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("pip install yfinance") from e
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    px = raw.dropna(axis=1, thresh=int(0.8 * len(raw))).ffill().dropna()
    return px


def load_csv(path: str) -> pd.DataFrame:
    """Precios desde CSV local (columna Date + una columna por ticker)."""
    px = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return px.sort_index().ffill().dropna()


def simulate_market(n_assets: int = 20, n_days: int = 1500,
                    seed: int = 20260704) -> pd.DataFrame:
    """Panel GBM multivariado correlacionado (SOLO tests/demos offline).

    Volatilidades heterogeneas (10%-60% anual) y primas de riesgo crecientes
    con la volatilidad (mu = rf + sharpe * sigma), para que exista una
    relacion riesgo-retorno explotable como en un mercado real.
    """
    rng = np.random.default_rng(seed)
    vols = np.linspace(0.10, 0.60, n_assets)
    mus = 0.02 + 0.35 * vols + rng.normal(0, 0.02, n_assets)
    corr = 0.35 * np.ones((n_assets, n_assets)) + 0.65 * np.eye(n_assets)
    L = np.linalg.cholesky(corr)
    dt = 1.0 / 252
    shocks = rng.standard_normal((n_days, n_assets)) @ L.T
    rets = (mus - 0.5 * vols ** 2) * dt + vols * np.sqrt(dt) * shocks
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"A{i+1:02d}" for i in range(n_assets)]
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=cols)
