"""Visualizaciones Okabe-Ito, fondo blanco, 300 dpi (estandar de la tesis)."""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#: Paleta Okabe-Ito (accesible para daltonismo), 8 colores = 8 perfiles.
OKABE_ITO: List[str] = ["#000000", "#E69F00", "#56B4E9", "#009E73",
                        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

__all__ = ["OKABE_ITO", "style", "plot_orness_ladder", "plot_frontier",
           "plot_adaptive_path"]


def style(ax):
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    return ax


def _save(fig, path: Optional[str]):
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_orness_ladder(profiles, path: Optional[str] = None):
    """Escalera de anclas: perfil vs orness (y latente z si existe)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [p.name for p in profiles]
    alphas = [p.alpha for p in profiles]
    ax.bar(names, alphas, color=OKABE_ITO, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Orness $\\alpha_k$")
    ax.set_ylim(0, 1)
    for i, a in enumerate(alphas):
        ax.text(i, a + 0.02, f"{a:.4f}", ha="center", fontsize=8)
    style(ax)
    ax.set_title("Profile anchors: orness ladder")
    plt.xticks(rotation=30, ha="right")
    _save(fig, path)
    return fig


def plot_frontier(results: Dict[str, object], path: Optional[str] = None):
    """Volatilidad esperada vs retorno esperado de las 8 carteras."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, (name, r) in enumerate(results.items()):
        ax.scatter(r.expected_vol, r.expected_return, s=90,
                   color=OKABE_ITO[i % 8], label=name, edgecolor="black",
                   linewidth=0.5, zorder=3)
    ax.set_xlabel("Expected volatility (ann.)")
    ax.set_ylabel("Expected return (ann.)")
    ax.set_title("Eight profile portfolios: risk-return alignment")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    style(ax)
    _save(fig, path)
    return fig


def plot_adaptive_path(history: List[dict], path: Optional[str] = None):
    """Trayectoria del latente z y migraciones de perfil de un inversor."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    z = [h["z_before"] for h in history] + [history[-1]["z_after"]]
    ax.plot(range(len(z)), z, marker="o", color=OKABE_ITO[5], linewidth=1.5)
    from .latent import phi_inv
    for j in range(1, 8):
        ax.axhline(phi_inv(j / 8), color="gray", linestyle=":", linewidth=0.7)
    for i, h in enumerate(history):
        if h["migrated"]:
            ax.axvline(i + 1, color=OKABE_ITO[6], alpha=0.5, linewidth=1)
    ax.set_xlabel("Investment horizon (cycle)")
    ax.set_ylabel("Latent risk appetite $z_t$")
    ax.set_title("Adaptive recalibration path (octile borders dotted)")
    style(ax)
    _save(fig, path)
    return fig
