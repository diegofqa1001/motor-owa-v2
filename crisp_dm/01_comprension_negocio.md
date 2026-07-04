# Fase 1 — Comprensión del negocio (CRISP-DM)

**Problema.** Los formularios de perfilamiento de la industria ("conservador /
moderado / arriesgado") son estáticos: no capturan la naturaleza dinámica del
apetito de riesgo ni la incertidumbre no probabilística de los mercados de
renta variable (anteproyecto, Planteamiento del problema).

**Objetivo de negocio.** Recomendar a cada inversor un portafolio de renta
variable coherente con su perfil conductual de riesgo, y **recalibrar** el
perfil cuando la experiencia del inversor (resultados vs. expectativas) lo
modifique (Obj. general + Obj. 3 del anteproyecto).

**Criterios de éxito.**
1. Ocho carteras (una por perfil) cuya volatilidad crece monótonamente con el
   orness del perfil: Spearman(orness, σ) = +1 (coherencia conductual).
2. Recalibración automática del perfil al cierre de cada horizonte con
   trazabilidad completa (auditable).
3. Métricas de validación interna declaradas en el anteproyecto: RMSE, MAE,
   MAPE, NDCG@k, MRR, consistencia ordinal, partición 70-20-10.

**Restricciones.** Incertidumbre no probabilística (agregación OWA, no
utilidad esperada); explicabilidad (ranking multicriterio trazable);
reproducibilidad (open science, semillas fijas).
