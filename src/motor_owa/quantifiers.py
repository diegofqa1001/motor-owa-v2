"""Cuantificador RIM Q(r)=r^beta y resolucion exacta de beta para un orness.

El anteproyecto (Obj. 3) define la funcion generadora de pesos OWA como el
cuantificador linguistico Regular Increasing Monotone (RIM) de la familia
potencia Q(r) = r^beta (Yager, 1996):

    w_j = Q(j/n) - Q((j-1)/n) = (j/n)^beta - ((j-1)/n)^beta

Para n -> infinito, orness -> 1/(beta+1); para n finito difiere, por lo que
beta se resuelve NUMERICAMENTE (biseccion geometrica sobre una funcion
estrictamente monotona) para materializar el orness del perfil de forma
exacta con el numero de criterios que use el motor (n=4) o el cuestionario
(n=7). Mismo orness, distinto beta segun n.
"""
from __future__ import annotations

import numpy as np

from .owa import orness, validate_weights

__all__ = ["rim_weights", "exponent_for_orness", "weights_for_orness",
           "orness_limit"]


def rim_weights(n: int, beta: float) -> np.ndarray:
    """Pesos OWA del cuantificador RIM Q(r)=r^beta (beta>0).

    beta < 1 -> orness > 0.5 (optimista); beta = 1 -> uniforme (0.5);
    beta > 1 -> orness < 0.5 (pesimista).
    """
    if n < 1:
        raise ValueError("n debe ser >= 1.")
    if beta <= 0:
        raise ValueError("beta debe ser positivo.")
    if n == 1:
        return np.array([1.0])
    j = np.arange(1, n + 1, dtype=float)
    w = (j / n) ** beta - ((j - 1) / n) ** beta
    return validate_weights(w)


def orness_limit(beta: float) -> float:
    """Limite continuo del orness del RIM potencia: 1/(beta+1)."""
    return 1.0 / (1.0 + float(beta))


def exponent_for_orness(n: int, target: float, tol: float = 1e-12,
                        max_iter: int = 300) -> float:
    """beta tal que orness(rim_weights(n, beta)) = target (biseccion).

    orness es estrictamente decreciente en beta, de ~1 (beta->0) a ~0
    (beta->inf): la raiz existe y es unica para target en (0,1).
    """
    target = float(np.clip(target, 1e-9, 1 - 1e-9))
    if n == 1:
        return 1.0
    lo, hi = 1e-9, 1e9
    f = lambda b: orness(rim_weights(n, b)) - target
    if f(lo) < 0:
        return lo
    if f(hi) > 0:
        return hi
    for _ in range(max_iter):
        mid = float(np.sqrt(lo * hi))  # biseccion geometrica (beta cruza ordenes)
        fm = f(mid)
        if abs(fm) < tol:
            return mid
        if fm > 0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def weights_for_orness(n: int, target: float) -> np.ndarray:
    """Pesos OWA (RIM) que materializan exactamente un orness objetivo."""
    return rim_weights(n, exponent_for_orness(n, target))
