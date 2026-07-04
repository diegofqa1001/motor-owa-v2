# Fase 5 — Evaluación / validación interna (CRISP-DM, Obj. 4)

**Protocolo** (anteproyecto): simulaciones con datos históricos, partición
70-20-10, sin usuarios reales.

**Métricas implementadas** (`validation.py`, reportadas por
`engine.panel_backtest`):

| Métrica | Qué mide | Implementación |
|---|---|---|
| RMSE / MAE / MAPE | error entre scores de verificación y validación | `rmse, mae, mape` |
| NDCG@k | calidad del ranking top-k | `ndcg_at_k` |
| MRR | posición del primer acierto | `mrr` |
| Consistencia ordinal | pares de activos que conservan orden | `ordinal_consistency` |
| Coherencia conductual | Spearman(orness, σ realizada) | `coherence_spearman` |

**Resultados de referencia (panel sintético, semilla 20260704):**
coherencia orness-vol = **+1.000**; NDCG@k ≈ 0.90–0.96; consistencia
ordinal ≈ 0.77–0.89. El retorno realizado **no** es monótono en muestras
cortas: el premio al riesgo es potencial, no garantizado (consistente con el
hallazgo del Artículo 3: la dimensión robusta de coherencia es la volatilidad).

**Criterios adicionales declarados:** estabilidad ordinal entre particiones,
sensibilidad al cambio de perfil (matriz de migraciones del módulo
adaptativo), reducción del riesgo máximo para el perfil conservador.
