"""Validacion interna (Obj. 4 del anteproyecto): metricas y particiones.

Implementa EXACTAMENTE lo declarado: RMSE, MAE, MAPE, consistencia ordinal,
NDCG@k, MRR, particion 70-20-10, mas la metrica de coherencia conductual
(Spearman orness-volatilidad) que es el eje del Articulo 3.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["rmse", "mae", "mape", "ndcg_at_k", "mrr", "ordinal_consistency",
           "split_70_20_10", "coherence_spearman", "spearman"]


def _pair(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size == 0:
        raise ValueError("Las series deben tener igual longitud > 0.")
    return a, b


def rmse(y_true, y_pred) -> float:
    a, b = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true, y_pred) -> float:
    a, b = _pair(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def mape(y_true, y_pred, eps: float = 1e-9) -> float:
    a, b = _pair(y_true, y_pred)
    return float(np.mean(np.abs((a - b) / np.maximum(np.abs(a), eps))) * 100)


def spearman(a, b) -> float:
    """Correlacion de Spearman sin scipy (rangos promedio en empates)."""
    a, b = _pair(a, b)
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.mean(ra * rb))


def ndcg_at_k(relevance_true: Sequence[float], order_pred: Sequence[int],
              k: int = 10) -> float:
    """NDCG@k: calidad del ranking predicho frente a la relevancia real.

    relevance_true[i] = relevancia del item i; order_pred = indices de los
    items en el orden recomendado (mejor primero).
    """
    rel = np.asarray(relevance_true, dtype=float)
    order = np.asarray(order_pred, dtype=int)[:k]
    gains = rel[order]
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(rel)[::-1][:k]
    idcg = float(np.sum(ideal * (1.0 / np.log2(np.arange(2, ideal.size + 2)))))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevant: Sequence[int], order_pred: Sequence[int]) -> float:
    """Mean Reciprocal Rank: 1/posicion del primer item relevante."""
    rel = set(int(i) for i in relevant)
    for pos, item in enumerate(order_pred, start=1):
        if int(item) in rel:
            return 1.0 / pos
    return 0.0


def ordinal_consistency(rank_a: Sequence[int], rank_b: Sequence[int]) -> float:
    """Proporcion de pares de activos que conservan su orden relativo."""
    a = np.asarray(rank_a, dtype=float)
    b = np.asarray(rank_b, dtype=float)
    if a.size != b.size or a.size < 2:
        raise ValueError("Se requieren >= 2 items con igual longitud.")
    n, keep, tot = a.size, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += 1
            if np.sign(a[i] - a[j]) == np.sign(b[i] - b[j]):
                keep += 1
    return keep / tot


def split_70_20_10(n: int) -> Tuple[slice, slice, slice]:
    """Particion temporal 70-20-10 (entrenamiento, verificacion, validacion)."""
    if n < 10:
        raise ValueError("Se requieren >= 10 observaciones.")
    i1, i2 = int(0.70 * n), int(0.90 * n)
    return slice(0, i1), slice(i1, i2), slice(i2, n)


def coherence_spearman(orness_values: Sequence[float],
                       realized_metric: Sequence[float]) -> float:
    """Coherencia conductual: Spearman(orness de los perfiles, metrica realizada).

    +1 = perfectamente coherente (el riesgo crece con el apetito declarado);
    -1 = invertida (hallazgo de la via de criterios pura, Articulo 3).
    """
    return spearman(orness_values, realized_metric)
