"""Los 8 perfiles conductuales de riesgo (Articulo 2) con anclas derivadas.

Cada perfil expone: nombre, indice k, latente z_k, orness alpha_k, y los
vectores de pesos OWA para n arbitrario (via cuantificador RIM). El motor
usa n=4 (criterios financieros del anteproyecto); el instrumento usa n=7
(dimensiones conductuales). El orness se CONSERVA entre ambos; beta cambia.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import (Anchors, ARTICULO2_ORNESS, DIMENSIONS as _DIMS,
                     PROFILE_NAMES)
from .latent import octile_orness, octile_z, classify_orness, classify_z
from .quantifiers import exponent_for_orness, weights_for_orness

DIMENSIONS: List[str] = list(_DIMS.keys())

__all__ = ["PROFILE_NAMES", "DIMENSIONS", "Profile", "all_profiles",
           "get_profile", "profile_orness"]


def profile_orness(k: int, anchors: Anchors = Anchors.OCTILES) -> float:
    """Orness del perfil k (1..8) segun la parametrizacion elegida."""
    if anchors == Anchors.OCTILES:
        return octile_orness(k)
    return ARTICULO2_ORNESS[k - 1]


@dataclass(frozen=True)
class Profile:
    """Perfil conductual de riesgo con ancla matematica.

    Atributos
    ---------
    k : int            indice 1 (Guardian) .. 8 (Visionary)
    name : str         nombre canonico del Articulo 2
    z : float          latente representativo (solo anclas OCTILES; NaN si A2)
    alpha : float      orness del perfil
    """
    k: int
    name: str
    z: float
    alpha: float

    def beta(self, n: int) -> float:
        """Exponente RIM que materializa el orness del perfil con n criterios."""
        return exponent_for_orness(n, self.alpha)

    def weights(self, n: int) -> np.ndarray:
        """Vector de pesos OWA del perfil para n criterios."""
        return weights_for_orness(n, self.alpha)

    @property
    def is_conservative(self) -> bool:
        return self.alpha < 0.5


def all_profiles(anchors: Anchors = Anchors.OCTILES) -> List[Profile]:
    """Los 8 perfiles ordenados de conservador a agresivo."""
    out = []
    for k in range(1, 9):
        a = profile_orness(k, anchors)
        z = octile_z(k) if anchors == Anchors.OCTILES else float("nan")
        out.append(Profile(k=k, name=PROFILE_NAMES[k - 1], z=z, alpha=a))
    return out


def get_profile(name_or_k, anchors: Anchors = Anchors.OCTILES) -> Profile:
    """Busca un perfil por nombre (str) o indice (int 1..8)."""
    profs = all_profiles(anchors)
    if isinstance(name_or_k, int):
        if not 1 <= name_or_k <= 8:
            raise ValueError("indice de perfil fuera de 1..8")
        return profs[name_or_k - 1]
    for p in profs:
        if p.name.lower() == str(name_or_k).lower():
            return p
    raise KeyError(f"Perfil desconocido: {name_or_k!r}")
