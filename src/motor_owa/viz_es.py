"""Rotulado en espanol para las figuras de la monografia.

Coma decimal en los ejes (formateador), nombres de perfiles en espanol,
paleta Okabe-Ito, fondo blanco y 300 dpi.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, PercentFormatter

from .viz import OKABE_ITO  # noqa: F401  (reexportada)

NOMBRES_ES = {"Guardian": "Guardián", "Sentinel": "Centinela",
              "Pragmatist": "Pragmático", "Analyst": "Analista",
              "Strategist": "Estratega", "Adventurer": "Aventurero",
              "Innovator": "Innovador", "Visionary": "Visionario"}

COMPARADORES_ES = {"1/N": "1/N", "MinVar": "Mínima varianza",
                   "MaxSharpe": "Máximo Sharpe", "MLP": "Red neuronal (MLP)",
                   "ANFIS": "ANFIS"}

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.size": 10,
                     "axes.unicode_minus": True})


class ComaFormatter(ScalarFormatter):
    """ScalarFormatter con coma decimal."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


class ComaPercentFormatter(PercentFormatter):
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def coma(x: float, dec: int = 2, signo: bool = False) -> str:
    """Numero con coma decimal (y signo explicito si signo=True)."""
    s = f"{x:+.{dec}f}" if signo else f"{x:.{dec}f}"
    return s.replace(".", ",").replace("-", "−")


def ejes_coma(ax, x: bool = True, y: bool = True):
    if x:
        ax.xaxis.set_major_formatter(ComaFormatter(useOffset=False))
    if y:
        ax.yaxis.set_major_formatter(ComaFormatter(useOffset=False))
    return ax


def estilo(ax):
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    return ax


def guardar(fig, path: str):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
