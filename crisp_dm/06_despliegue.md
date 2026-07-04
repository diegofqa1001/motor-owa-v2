# Fase 6 — Despliegue conceptual (CRISP-DM)

- **Paquete instalable** `motor-owa-v2` (pip install -e .), API pública en
  `motor_owa/__init__.py`.
- **Demo de un comando:** `python scripts/run_demo.py [--market co|us]`
  → figuras Okabe-Ito 300 dpi (fondo blanco) + CSV de resultados.
- **Reproducibilidad:** semilla global (`EngineConfig.seed`), tests
  `pytest` (41 casos), sin dependencias más allá de numpy/pandas/matplotlib
  (yfinance opcional).
- **Trazabilidad:** cada recomendación expone scores, activos, pesos, σ
  objetivo/alcanzada y θ; cada recalibración registra sorpresa, z y perfil
  antes/después.
- **Línea futura** (fuera del alcance del anteproyecto): despliegue en
  entorno real (API REST / Streamlit), validación con usuarios finales
  previa aprobación de comité de ética.
