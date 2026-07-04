# Fase 2 — Comprensión de los datos (CRISP-DM)

**Fuentes.** Precios ajustados diarios vía Yahoo Finance (`yfinance`):
- **Colombia (BVC)** — mercado declarado del anteproyecto: 17 emisores líquidos
  (`config.TICKERS_CO`).
- **EE. UU.** — universo de robustez (profundidad de mercado): 25 blue chips
  (`config.TICKERS_US`).
- Alternativas: CSV local (`data.load_csv`) y panel sintético GBM
  (`data.simulate_market`) **solo** para pruebas unitarias y demos offline.

**Variables por activo (m = 4, anteproyecto Obj. 3):**

| Criterio | Definición | Codificación |
|---|---|---|
| Rentabilidad | retorno medio anualizado (ventana 126 d) | directa |
| Baja volatilidad | desv. típica anualizada | inversa |
| Baja caída máxima | maximum drawdown de la ventana | inversa |
| Liquidez/estabilidad | volumen medio o estabilidad de la vol | directa |

**Calidad.** Se exige ≥ 80 % de datos válidos por columna; activos deslistados
se excluyen (p. ej., PFBCOLOM.CL en el Artículo 3). El mercado CO es pequeño y
correlacionado: se documenta como limitación (perfiles pueden solaparse).
