# Fase 3 — Preparación de los datos (CRISP-DM)

1. **Limpieza:** forward-fill de huecos, eliminación de activos con < 80 % de
   historia, orden temporal estricto (sin look-ahead).
2. **Ventanas rodantes:** cada evaluación usa exclusivamente la ventana
   `(t - lookback, t]` (lookback = 126 días hábiles por defecto).
3. **Normalización [0,1]:** min-max en sección cruzada por criterio y fecha
   (anteproyecto: "normalización de los criterios en el intervalo [0,1] para
   garantizar compatibilidad agregativa"). 1 = mejor del universo en ese
   criterio y fecha; serie constante → 0.5.
4. **Partición 70-20-10** de la rejilla temporal de evaluación:
   entrenamiento / verificación intermedia / validación final
   (`validation.split_70_20_10`), respetando el orden cronológico.

Módulos: `criteria.py` (matriz de decisión), `data.py` (carga y simulación).
