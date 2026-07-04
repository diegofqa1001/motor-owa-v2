# motor-owa-v2 — Motor adaptativo de recomendación de portafolios

**Versión 2 (CRISP-DM)** del motor de recomendación de la tesis doctoral
*Modelo adaptativo de recomendación para el diseño de portafolios de inversión
en renta variable bajo incertidumbre, mediante el operador OWA y perfiles
conductuales de riesgo* (Universidad Nacional de Colombia, Sede Manizales).

> Este paquete es independiente del motor v1 (`owa-adaptive`) y no lo modifica.

## Qué hace

1. **Clasifica** al inversor en uno de **8 perfiles conductuales**
   (Guardian → Visionary) a partir de 7 dimensiones conductuales.
   Las anclas de los perfiles están **derivadas matemáticamente**:
   latente z ~ N(0,1), octiles, `orness_k = Φ(z_k) = (2k−1)/16`.
2. **Recomienda** la cartera del perfil: selección multicriterio OWA
   (`Q(r)=r^β`, m = 4 criterios del anteproyecto) **estratificada por
   volatilidad**, con volatilidad objetivo creciente en el orness
   → **8 carteras, coherencia riesgo-perfil = +1 por construcción**.
3. **Recalibra** el perfil al cierre de cada horizonte con **el decisor al
   mando (v2.1)**: el motor muestra el resultado frente a la proyección
   inicial y el inversor **vuelve a responder el cuestionario**; su
   declaración reclasifica el perfil. En paralelo el modelo predice la
   reacción "lógica" (sorpresa acotada, λ = 2.25) y registra la **brecha
   emocional ε = z_declarada − z_modelo**, que valida que la decisión no
   es puramente lógica. Sin declaración, opera el canal automático.
4. **Valida** internamente según el anteproyecto: RMSE, MAE, MAPE, NDCG@k,
   MRR, consistencia ordinal, partición 70-20-10, coherencia Spearman.

## Estructura (CRISP-DM)

```
crisp_dm/            fases 1-6 documentadas (negocio → despliegue)
src/motor_owa/
  config.py          anclas (octiles | articulo2), parámetros, universos CO/US
  latent.py          espacio latente z: Φ, octiles, clasificación, puente 7D→z
  owa.py             operador OWA, orness, andness
  quantifiers.py     RIM Q(r)=r^β, β exacto para un orness (bisección)
  profiles.py        los 8 perfiles con anclas derivadas
  criteria.py        matriz de decisión [activos × 4] normalizada [0,1]
  portfolio.py       8 carteras con volatilidad objetivo (coherencia +1)
  adaptive.py        sorpresa → actualización latente → migración de perfil
  elicitation.py     cuestionario operativo, canal declarado, brecha emocional ε
  engine.py          ciclo completo + backtest de panel 70-20-10
  validation.py      RMSE, MAE, MAPE, NDCG@k, MRR, consistencia ordinal
  data.py            yfinance CO/US, CSV, panel sintético (solo tests)
  viz.py             figuras Okabe-Ito, fondo blanco, 300 dpi
scripts/run_demo.py  demo de un comando
tests/               49 pruebas pytest
```

## Inicio rápido

```bash
pip install -e ".[data,dev]"
pytest                      # 49 tests
python scripts/run_demo.py            # demo offline (panel sintético)
python scripts/run_demo.py --market co   # Colombia (BVC, anteproyecto)
python scripts/run_demo.py --market us   # EE. UU. (robustez)
```

```python
from motor_owa import EngineConfig, InvestorState
from motor_owa.engine import RecommendationEngine
from motor_owa.data import load_yfinance, TICKERS_CO

px = load_yfinance(TICKERS_CO)
eng = RecommendationEngine(px, EngineConfig())
inv = InvestorState.from_profile("Pragmatist")
ciclo = eng.run_cycle(inv, t=len(px) - 200)   # recomienda, invierte, recalibra
print(ciclo.portfolio.weights, ciclo.migrated, inv.profile.name)
```

## Las dos parametrizaciones de anclas

| k | Perfil | Octiles Φ(z) (default) | Artículo 2 |
|---|---|---|---|
| 1 | Guardian | 0.0625 | 0.158 |
| 2 | Sentinel | 0.1875 | 0.257 |
| 3 | Pragmatist | 0.3125 | 0.503 |
| 4 | Analyst | 0.4375 | 0.600 |
| 5 | Strategist | 0.5625 | 0.647 |
| 6 | Adventurer | 0.6875 | 0.693 |
| 7 | Innovator | 0.8125 | 0.738 |
| 8 | Visionary | 0.9375 | 0.865 |

`EngineConfig(anchors=Anchors.ARTICULO2)` reproduce los valores publicados.

## Licencia

MIT (código). Contenido y figuras: CC-BY-4.0.
