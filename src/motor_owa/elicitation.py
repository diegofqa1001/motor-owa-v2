"""Re-elicitacion declarada: el DECISOR gobierna la adaptacion (ajuste v2.1).

Idea (Diego Quintero, 2026-07-04): la adaptabilidad no debe ser solo una
regla automatica del modelo. Como en la practica bancaria, el motor
PREGUNTA (cuestionario de 7 dimensiones), clasifica, recomienda; el
inversor define VALOR y HORIZONTE; al cierre, el motor muestra el
resultado frente a la proyeccion inicial y el inversor VUELVE A RESPONDER
el cuestionario con su sentimiento e informacion nueva. Esa respuesta
declarada reclasifica al inversor.

El diseno permite VALIDAR que la decision no es puramente logica:

    z_modelo   = z_t + kappa*Lambda(s)*tanh(s)     (prediccion "racional
                 acotada" del modelo: reaccion proporcional a la sorpresa)
    z_declarada = puente(respuestas nuevas)         (lo que la persona dice)
    epsilon     = z_declarada - z_modelo            (BRECHA EMOCIONAL)

Si las decisiones fueran puramente logicas (dado el modelo), epsilon ~ 0.
Un epsilon sistematicamente distinto de cero, correlacionado con el signo
de la sorpresa o con la racha reciente, es evidencia cuantificable del
componente emocional. Ademas, la asimetria de perdida lambda deja de ser
un supuesto: se puede ESTIMAR con las respuestas declaradas
(lambda_hat = |pendiente en perdidas| / |pendiente en ganancias|).

Sin datos de campo (que requieren aval etico), la validacion interna del
Obj. 4 usa DECISORES SINTETICOS (simulate_declared_scores) con un
generador emocional controlado: si el pipeline recupera los parametros
sembrados, el mecanismo de medicion queda validado internamente.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import DIMENSIONS
from .latent import latent_from_dimensions, phi, phi_inv

__all__ = ["QUESTIONNAIRE", "declared_z", "scores_from_z",
           "simulate_declared_scores", "emotional_gap_metrics"]

#: Cuestionario operativo (tipo banca), escala Likert 1-5.
#: (clave de dimension, pregunta, ancla 1, ancla 5)
QUESTIONNAIRE: List[Tuple[str, str, str, str]] = [
    ("risk_tolerance",
     "¿Que tan dispuesto(a) esta a aceptar caidas temporales de su inversion "
     "a cambio de una posible mayor ganancia final?",
     "Nada dispuesto", "Totalmente dispuesto"),
    ("loss_aversion",
     "¿Cuanto le molesta perder dinero, comparado con lo que le alegra ganar "
     "la misma cantidad?",
     "Me molesta mucho mas", "Me da igual"),
    ("financial_self_efficacy",
     "¿Que tan capaz se siente de elegir por si mismo(a) en que invertir?",
     "Nada capaz", "Totalmente capaz"),
    ("ambiguity_tolerance",
     "¿Que tan comodo(a) invierte cuando la informacion es incompleta?",
     "Nada comodo", "Totalmente comodo"),
    ("investment_horizon",
     "¿Por cuanto tiempo planea mantener esta inversion? "
     "(1: menos de 1 año ... 5: mas de 10 años)",
     "Menos de 1 año", "Mas de 10 años"),
    ("emotional_regulation",
     "Cuando sus inversiones caen fuerte en pocos dias, ¿logra evitar "
     "decisiones impulsivas como vender de inmediato?",
     "Nunca lo logro", "Siempre lo logro"),
    ("social_influence",
     "¿Suele invertir en lo que ve que esta invirtiendo la mayoria?",
     "Siempre", "Nunca"),
]

# Nota de codificacion: loss_aversion y social_influence se preguntan en
# sentido inverso (5 = menos aversion / menos influencia), de modo que la
# direccion efectiva del puntaje bruto es +1 para TODAS las preguntas tal
# como estan redactadas. El diccionario de direcciones del puente usa el
# signo teorico de la DIMENSION; aqui pasamos las respuestas ya orientadas.
_ORIENTED_DIRECTIONS: Dict[str, float] = {k: 1.0 for k in DIMENSIONS}


def declared_z(scores: Dict[str, float]) -> float:
    """Latente declarada a partir de las respuestas Likert 1-5 orientadas."""
    return latent_from_dimensions(scores, _ORIENTED_DIRECTIONS,
                                  likert_min=1.0, likert_max=5.0)


def scores_from_z(z: float, rng: Optional[np.random.Generator] = None,
                  noise_sd: float = 0.0) -> Dict[str, float]:
    """Inversa aproximada del puente: respuestas coherentes con una latente z.

    u = 2*Phi(z) - 1 en [-1,1]; cada respuesta x_j = 3 + 2u (+ ruido),
    recortada a [1,5]. Con noise_sd > 0 simula inconsistencia del decisor.
    """
    rng = rng or np.random.default_rng()
    u = 2.0 * phi(z) - 1.0
    out = {}
    for k in DIMENSIONS:
        x = 3.0 + 2.0 * u
        if noise_sd > 0:
            x += rng.normal(0.0, noise_sd)
        out[k] = float(np.clip(x, 1.0, 5.0))
    return out


def simulate_declared_scores(z_prev: float, surprise: float,
                             kappa: float, loss_lambda: float,
                             emotion_gain: float = 0.0,
                             sentiment: float = 0.0,
                             noise_sd: float = 0.15,
                             rng: Optional[np.random.Generator] = None
                             ) -> Dict[str, float]:
    """Decisor sintetico para la validacion interna (Obj. 4).

    Genera las respuestas que daria un inversor cuya latente se movio por
    (i) el canal "racional acotado" del modelo y (ii) un empuje EMOCIONAL
    controlado (emotion_gain * sentiment). Con emotion_gain = 0 el decisor
    es puramente logico y la brecha esperada es ~0 (salvo ruido): ese es
    el control negativo del experimento.
    """
    rng = rng or np.random.default_rng()
    lam = loss_lambda if surprise < 0 else 1.0
    z_rational = z_prev + kappa * lam * float(np.tanh(surprise))
    z_emotional = z_rational + emotion_gain * float(sentiment)
    return scores_from_z(z_emotional, rng=rng, noise_sd=noise_sd)


def emotional_gap_metrics(history: Sequence[dict],
                          z_bound: Optional[float] = None) -> Dict[str, float]:
    """Metricas de validacion del componente emocional sobre la bitacora.

    Requiere ciclos con re-elicitacion declarada (epsilon registrado).
    ``z_bound`` (opcional): CENSURA los ciclos donde la latente toca el
    rango saturado del instrumento (|z| >= z_bound). El cuestionario es una
    escala acotada: cerca de sus extremos la declaracion no puede moverse
    mas y los estimadores se atenuan; censurar el tramo saturado es el
    tratamiento estandar de censura y se reporta n junto a cada metrica.
    Devuelve:
      - mean_gap / mean_abs_gap: sesgo y magnitud de la brecha emocional.
      - corr_gap_surprise: correlacion brecha-sorpresa (0 si el modelo ya
        explica toda la reaccion; != 0 si hay reaccion emocional residual).
      - lambda_hat: asimetria de perdida ESTIMADA de las declaraciones,
        Delta_z_declarado/tanh(s) promedio en perdidas vs ganancias.
      - n: numero de ciclos con declaracion (tras censura).
    """
    eps, s_all = [], []
    slope_loss, slope_gain = [], []
    for h in history:
        if "epsilon" not in h or h["epsilon"] is None:
            continue
        if z_bound is not None and (abs(float(h["z_before"])) >= z_bound
                                    or abs(float(h["z_after"])) >= z_bound):
            continue
        s = float(h["surprise"])
        eps.append(float(h["epsilon"]))
        s_all.append(s)
        dz_dec = float(h["z_after"]) - float(h["z_before"])
        g = float(np.tanh(s))
        if abs(g) > 1e-6:
            (slope_loss if s < 0 else slope_gain).append(dz_dec / g)
    if not eps:
        return {"n": 0}
    eps_a, s_a = np.array(eps), np.array(s_all)
    corr = 0.0
    if len(eps_a) > 2 and eps_a.std() > 1e-12 and s_a.std() > 1e-12:
        corr = float(np.corrcoef(eps_a, s_a)[0, 1])
    lam_hat = float("nan")
    if slope_loss and slope_gain:
        den = float(np.mean(slope_gain))
        if abs(den) > 1e-9:
            lam_hat = float(np.mean(slope_loss)) / den
    return {"n": len(eps_a), "mean_gap": float(eps_a.mean()),
            "mean_abs_gap": float(np.abs(eps_a).mean()),
            "corr_gap_surprise": corr, "lambda_hat": lam_hat}
