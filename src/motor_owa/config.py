"""Configuracion del motor v2: anclas de perfiles y parametros globales."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class Anchors(str, Enum):
    """Parametrizacion de las anclas de orness de los 8 perfiles.

    OCTILES  -- derivacion matematica (default): latente z ~ N(0,1) dividida
                en octiles; orness_k = Phi(z_k) = (2k-1)/16, k = 1..8.
                Rango [0.0625, 0.9375], simetrico, neutral en 0.5.
                (Respuesta metodologica a Leon-Castro, 29-jun-2026.)
    ARTICULO2 -- anclas empiricas publicadas en el Articulo 2 (en revision):
                centroides difusos -> orness en [0.158, 0.865], no
                equiespaciado. Se conserva para reproducir el articulo.
    """
    OCTILES = "octiles"
    ARTICULO2 = "articulo2"


#: Orness del Articulo 2 (Tabla 4), en orden Guardian -> Visionary.
ARTICULO2_ORNESS: List[float] = [0.158, 0.257, 0.503, 0.600,
                                 0.647, 0.693, 0.738, 0.865]

#: Nombres canonicos de los 8 perfiles (Articulo 2), conservador -> agresivo.
PROFILE_NAMES: List[str] = [
    "Guardian", "Sentinel", "Pragmatist", "Analyst",
    "Strategist", "Adventurer", "Innovator", "Visionary",
]

#: 7 dimensiones conductuales del Articulo 2 (Tabla 4). direction indica el
#: signo teorico sobre el apetito de riesgo (+ sube orness, - baja).
DIMENSIONS: Dict[str, float] = {
    "risk_tolerance":        +1.0,  # D1  tolerancia al riesgo
    "loss_aversion":         -1.0,  # D5  aversion a la perdida (codif. inversa)
    "financial_self_efficacy": +1.0,  # D4  autoeficacia financiera
    "ambiguity_tolerance":   +1.0,  # D10 tolerancia a la ambiguedad
    "investment_horizon":    +1.0,  # D8  horizonte de inversion
    "emotional_regulation":  +1.0,  # D7  regulacion emocional
    "social_influence":      -1.0,  # D12 influencia social percibida (inversa)
}


@dataclass
class EngineConfig:
    """Parametros del motor.

    Atributos
    ---------
    anchors : Anchors
        Parametrizacion de anclas de orness (default: OCTILES).
    n_criteria : int
        Numero de criterios financieros por activo (anteproyecto: m=4).
    top_n : int
        Numero de activos que componen cada cartera recomendada.
    max_weight : float
        Tope de peso por activo (diversificacion; robustez Art. 3).
    lookback : int
        Ventana (dias habiles) para estimar criterios y covarianza.
    horizon : int
        Horizonte de inversion (dias habiles) entre recalibraciones.
    kappa : float
        Ganancia de aprendizaje de la actualizacion latente (0 = estatico).
    loss_lambda : float
        Asimetria de aversion a la perdida en la actualizacion latente
        (Tversky & Kahneman, 1992 estiman ~2.25).
    tc_bps : float
        Costos de transaccion en puntos basicos por rebalanceo.
    seed : int
        Semilla global de reproducibilidad.
    """
    anchors: Anchors = Anchors.OCTILES
    n_criteria: int = 4
    top_n: int = 10
    max_weight: float = 0.30
    lookback: int = 126
    horizon: int = 63
    kappa: float = 0.25
    loss_lambda: float = 2.25
    tc_bps: float = 10.0
    seed: int = 20260704


#: Universos de referencia (Articulo 3). CO: BVC via Yahoo Finance (.CL).
TICKERS_CO: List[str] = [
    "ECOPETROL.CL", "ISA.CL", "GEB.CL", "BCOLOMBIA.CL", "PFBCOLOM.CL",
    "GRUPOSURA.CL", "GRUPOARGOS.CL", "CEMARGOS.CL", "NUTRESA.CL",
    "EXITO.CL", "PROMIGAS.CL", "CELSIA.CL", "BOGOTA.CL", "CORFICOLCF.CL",
    "PFDAVVNDA.CL", "MINEROS.CL", "TERPEL.CL",
]

TICKERS_US: List[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V",
    "PG", "JNJ", "UNH", "HD", "MA", "XOM", "KO", "PEP", "WMT", "MCD",
    "CSCO", "ABT", "CVX", "MRK", "DIS", "VZ",
]
