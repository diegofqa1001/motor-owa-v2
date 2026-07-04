"""Operador OWA y medidas de actitud (Yager, 1988)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["validate_weights", "owa", "orness", "andness"]


def validate_weights(w: Sequence[float]) -> np.ndarray:
    """Valida un vector de pesos OWA: no negativo y suma 1 (renormaliza)."""
    w = np.asarray(w, dtype=float).ravel()
    if w.size == 0:
        raise ValueError("El vector de pesos esta vacio.")
    if np.any(w < -1e-12):
        raise ValueError("Los pesos OWA no pueden ser negativos.")
    s = w.sum()
    if s <= 0:
        raise ValueError("La suma de pesos debe ser positiva.")
    return np.clip(w / s, 0.0, 1.0)


def owa(values: Sequence[float], weights: Sequence[float]) -> float:
    """OWA(a; w) = sum_j w_j * a_(j), con a_(1) >= a_(2) >= ... (descendente)."""
    a = np.sort(np.asarray(values, dtype=float).ravel())[::-1]
    w = validate_weights(weights)
    if a.size != w.size:
        raise ValueError("values y weights deben tener igual longitud.")
    return float(np.dot(w, a))


def orness(weights: Sequence[float]) -> float:
    """orness(w) = (1/(n-1)) * sum_j (n-j) * w_j  en [0,1] (Yager, 1988).

    1 = maximo (OR, optimista); 0 = minimo (AND, pesimista); 0.5 = media.
    """
    w = validate_weights(weights)
    n = w.size
    if n == 1:
        return 0.5
    j = np.arange(1, n + 1)
    return float(np.dot((n - j), w) / (n - 1))


def andness(weights: Sequence[float]) -> float:
    """andness = 1 - orness."""
    return 1.0 - orness(weights)
