"""Espacio latente del apetito de riesgo: derivacion matematica de anclas.

Idea central (optimizacion v2, responde a la objecion de Leon-Castro):
el apetito de riesgo del inversor se modela como una variable latente
continua z ~ N(0,1) (convencion psicometrica estandar: rasgos latentes
estandarizados). Los 8 perfiles son los 8 OCTILES de esa latente, y el
ancla de cada perfil es el PUNTO MEDIO de probabilidad de su octil:

    p_k = (2k - 1) / 16,   k = 1..8
    z_k = Phi^{-1}(p_k)
    orness_k = Phi(z_k) = p_k

Es decir: el orness del perfil k ES la posicion de percentil del centro
de su octil. Nada se asigna por juicio: dado (i) latente normal estandar,
(ii) 8 grupos equiprobables y (iii) representante = mediana del grupo,
los ocho valores quedan determinados de forma unica:

    [0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375]

Propiedades: simetria perfecta alrededor de 0.5 (neutralidad), extremos
no degenerados (ni 0 ni 1) y equiespaciamiento exacto de 1/8.
Sin dependencia de scipy: Phi via math.erf, Phi^{-1} via NormalDist.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Dict, List, Sequence

_STD_NORMAL = NormalDist()

__all__ = ["phi", "phi_inv", "octile_orness", "octile_z", "classify_z",
           "classify_orness", "latent_from_dimensions"]


def phi(z: float) -> float:
    """CDF normal estandar Phi(z) = P(Z <= z) = (1 + erf(z/sqrt(2)))/2."""
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def phi_inv(p: float) -> float:
    """Inversa de la CDF normal estandar (funcion cuantil)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p debe estar en (0, 1).")
    return _STD_NORMAL.inv_cdf(float(p))


def octile_orness(k: int, n_profiles: int = 8) -> float:
    """Orness del perfil k (1-indexado): punto medio del octil, (2k-1)/(2K)."""
    if not 1 <= k <= n_profiles:
        raise ValueError(f"k debe estar en 1..{n_profiles}.")
    return (2 * k - 1) / (2 * n_profiles)


def octile_z(k: int, n_profiles: int = 8) -> float:
    """Valor latente representativo del perfil k: z_k = Phi^{-1}((2k-1)/2K)."""
    return phi_inv(octile_orness(k, n_profiles))


def classify_z(z: float, n_profiles: int = 8) -> int:
    """Perfil (1..K) al que pertenece un latente z: el octil que lo contiene.

    Los cortes entre perfiles son los cuantiles j/K de la N(0,1),
    j = 1..K-1. Ej.: K=8 -> cortes en Phi^{-1}(1/8), ..., Phi^{-1}(7/8).
    """
    p = phi(z)
    k = int(p * n_profiles) + 1
    return max(1, min(n_profiles, k))


def classify_orness(alpha: float, n_profiles: int = 8) -> int:
    """Perfil al que pertenece un orness dado (bin uniforme en [0,1])."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("orness debe estar en [0, 1].")
    k = int(alpha * n_profiles) + 1
    return max(1, min(n_profiles, k))


def latent_from_dimensions(scores: Dict[str, float],
                           directions: Dict[str, float],
                           likert_min: float = 1.0,
                           likert_max: float = 5.0) -> float:
    """Puente conductual -> latente z (clasificacion de un inversor nuevo).

    Cada dimension Likert x_j se lleva a [-1, 1] centrada en el punto
    neutro, se pondera por su direccion teorica d_j (+1 sube apetito,
    -1 lo baja) y se promedia:

        u = (1/J) * sum_j d_j * (2*(x_j - min)/(max - min) - 1)   en [-1,1]

    El agregado u se proyecta a la latente por el cuantil de una
    distribucion de referencia uniforme: z = Phi^{-1}((u+1)/2), acotando
    para evitar infinitos. Un inversor neutro (todas las respuestas en el
    punto medio) obtiene u = 0 -> z = 0 -> orness 0.5, y los extremos
    absolutos van a los octiles extremos, no a +-infinito.
    """
    if set(scores) != set(directions):
        raise ValueError("scores y directions deben tener las mismas claves.")
    span = likert_max - likert_min
    vals = []
    for key, d in directions.items():
        x = float(scores[key])
        if not likert_min <= x <= likert_max:
            raise ValueError(f"{key}={x} fuera de la escala [{likert_min},{likert_max}].")
        vals.append(math.copysign(1.0, d) * abs(d) * (2.0 * (x - likert_min) / span - 1.0))
    u = sum(vals) / sum(abs(d) for d in directions.values())
    # Proyeccion cuantilica acotada: evita Phi^{-1}(0) o Phi^{-1}(1).
    eps = 1.0 / 32.0   # medio octil: los extremos caen en el centro del octil extremo
    p = min(1.0 - eps, max(eps, (u + 1.0) / 2.0))
    return phi_inv(p)
