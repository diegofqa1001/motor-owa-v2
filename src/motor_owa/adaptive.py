"""Capa adaptativa: el inversor tambien es dinamico (recalibracion por cosecha).

Ciclo del motor (por horizonte de inversion h definido por el inversor):

  1. clasificar: cuestionario -> latente z -> perfil k (octil).
  2. recomendar: cartera del perfil con volatilidad objetivo sigma_k.
  3. invertir y esperar el horizonte h.
  4. cosechar: retorno realizado R_h vs expectativa (mu_h, sigma_h) ex-ante.
  5. recalibrar: la sorpresa estandarizada actualiza la latente z; si z
     cruza la frontera de su octil, el inversor MIGRA de perfil y su
     cartera se reconstruye en el siguiente ciclo.

Regla de actualizacion (optimizacion v2, reemplaza el ajuste directo sobre
alpha del motor v1):

    s_t   = (R_h - mu_h) / max(sigma_h, eps)        (sorpresa estandarizada)
    g(s)  = tanh(s / tau)                            (acotada, saturante)
    dz    = kappa * g(s)            si s >= 0        (grata sorpresa)
    dz    = kappa * lambda * g(s)   si s <  0        (aversion a la perdida)
    z_t+1 = clip(z_t + dz, -z_cap, +z_cap)

Justificacion: (i) el estado vive en el espacio latente z (donde el anclaje
es lineal y simetrico) y no en el orness (acotado en (0,1)), evitando
distorsiones de borde: Phi garantiza orness en (0,1) SIEMPRE; (ii) la
asimetria lambda ~ 2.25 implementa la aversion a la perdida de la teoria
prospectiva (Tversky & Kahneman, 1992): una perdida inesperada reduce el
apetito de riesgo mas de lo que una ganancia equivalente lo aumenta;
(iii) tanh impide que un unico periodo extremo dispare migraciones de mas
de ~1 octil (estabilidad, sin recategorizaciones erraticas).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import Anchors, EngineConfig
from .latent import classify_z, phi, phi_inv, octile_z
from .profiles import Profile, all_profiles, get_profile

Z_CAP = 3.0      # |z| maximo (mas alla del percentil 99.87 no hay informacion)
TAU = 1.0        # escala de la sorpresa (1 desviacion tipica ex-ante)

__all__ = ["InvestorState", "surprise", "update_latent",
           "harvest_and_recalibrate"]


def surprise(realized: float, expected: float, expected_vol: float,
             eps: float = 1e-9) -> float:
    """Sorpresa estandarizada del horizonte: (R - mu) / sigma."""
    return (float(realized) - float(expected)) / max(float(expected_vol), eps)


def update_latent(z: float, s: float, kappa: float = 0.25,
                  loss_lambda: float = 2.25, tau: float = TAU,
                  z_cap: float = Z_CAP) -> float:
    """Nueva latente tras una sorpresa s (asimetrica en perdidas)."""
    g = float(np.tanh(s / tau))
    dz = kappa * g if s >= 0 else kappa * loss_lambda * g
    return float(np.clip(z + dz, -z_cap, z_cap))


@dataclass
class InvestorState:
    """Estado dinamico del inversor: latente, perfil y trazabilidad."""
    z: float
    anchors: Anchors = Anchors.OCTILES
    history: List[dict] = field(default_factory=list)

    @property
    def k(self) -> int:
        return classify_z(self.z)

    @property
    def profile(self) -> Profile:
        return get_profile(self.k, self.anchors)

    @property
    def orness(self) -> float:
        """Orness continuo del inversor (no solo el ancla del octil)."""
        return phi(self.z)

    @classmethod
    def from_profile(cls, name_or_k, anchors: Anchors = Anchors.OCTILES
                     ) -> "InvestorState":
        p = get_profile(name_or_k, anchors)
        z0 = p.z if np.isfinite(p.z) else phi_inv(p.alpha)
        return cls(z=z0, anchors=anchors)


def harvest_and_recalibrate(state: InvestorState, realized: float,
                            expected: float, expected_vol: float,
                            cfg: Optional[EngineConfig] = None) -> InvestorState:
    """Cierra un horizonte: registra la cosecha y recalibra el perfil.

    Devuelve el MISMO objeto (mutado) para encadenar ciclos. Registra en
    ``state.history`` la trazabilidad completa (auditable): sorpresa,
    z antes/despues, perfil antes/despues y si hubo migracion.
    """
    cfg = cfg or EngineConfig()
    k_before, z_before = state.k, state.z
    s = surprise(realized, expected, expected_vol)
    state.z = update_latent(z_before, s, cfg.kappa, cfg.loss_lambda)
    state.history.append({
        "realized": realized, "expected": expected,
        "expected_vol": expected_vol, "surprise": s,
        "z_before": z_before, "z_after": state.z,
        "k_before": k_before, "k_after": state.k,
        "migrated": state.k != k_before,
    })
    return state
