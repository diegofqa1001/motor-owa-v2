# Fase 4 — Modelado adaptativo (CRISP-DM)

Arquitectura de cinco componentes (anteproyecto, Obj. 3), versión 2:

1. **Entrada de perfiles difusos → parámetro adaptativo.**
   Latente z ~ N(0,1); 8 perfiles = octiles; ancla del perfil k:
   `orness_k = Φ(z_k) = (2k−1)/16` (derivada, no asertada). El puente
   cuestionario → z es `latent.latent_from_dimensions` (7 dimensiones del
   Artículo 2 con dirección teórica). Parametrización alternativa:
   anclas empíricas del Artículo 2 (`Anchors.ARTICULO2`).

2. **Entrada de criterios financieros por acción.** Matriz de decisión
   [activos × 4] normalizada (fase 3).

3. **Agregación OWA.** Pesos del cuantificador RIM `Q(r) = r^β`;
   β se resuelve numéricamente para materializar el orness del perfil con
   n = 4 exacto (`quantifiers.exponent_for_orness`). Score OWA por activo
   y ranking por perfil (explicable y trazable).

4. **Módulo de recomendación (carteras coherentes).** Selección OWA
   **estratificada por volatilidad** + mezcla de dos fondos (defensivo /
   agresivo) con volatilidad objetivo
   `σ_k = σ_def + orness_k · (σ_agg − σ_def)`.
   Garantiza Spearman(orness, σ) = +1 **por construcción**, corrigiendo la
   inversión conductual de la vía de criterios pura (hallazgo del Art. 3).

5. **Capa de adaptabilidad — el decisor manda (v2.1).** Dos canales:

   - **Canal declarado (principal).** Como en la práctica bancaria, el
     motor PREGUNTA (cuestionario de 7 dimensiones, `elicitation.QUESTIONNAIRE`),
     clasifica y recomienda; el inversor define **valor** (`wealth`) y
     **tiempo** (`horizon`); al cierre el motor muestra la **evaluación
     resultado-vs-proyección** (`CycleRecord.projection/evaluation`) y el
     inversor **vuelve a responder** el cuestionario con su sentimiento e
     información nueva. Su declaración fija la nueva latente y reclasifica
     el perfil: la decisión es del decisor, no del modelo.
   - **Canal automático (predicción "lógica").** En paralelo, el modelo
     calcula su predicción: sorpresa `s = (R − μ)/σ` y actualización
     asimétrica `z_modelo = z + κ·Λ(s)·tanh(s)` (λ = 2.25, Tversky &
     Kahneman, 1992). Si no hay declaración, este canal opera solo.

   La **brecha emocional** `ε = z_declarada − z_modelo` queda registrada
   en cada ciclo: cuantifica cuánto de la reclasificación NO es explicable
   por la reacción "lógica" a la sorpresa — la validación de que la
   decisión es también emocional (`elicitation.emotional_gap_metrics`,
   con censura del tramo saturado del instrumento). Recategorización si z
   cruza la frontera de su octil; trazabilidad completa en
   `InvestorState.history`.
