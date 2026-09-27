"""Verificacion numerica del paso maximo de la regla de actualizacion (§6.3).

Regla (src/motor_owa/adaptive.py): dz = kappa*tanh(s) si s >= 0 y
dz = kappa*lambda*tanh(s) si s < 0, con kappa = 0,25 y lambda = 2,25, de
modo que |dz| < kappa*lambda = 0,5625 hacia la prudencia y |dz| < kappa =
0,25 hacia la audacia. El guion calcula:

  1. los cortes de octil en z, Phi^{-1}(j/8), y el ancho de cada octil
     interior y de cada par de octiles interiores contiguos;
  2. para cada perfil (latente en su ancla, Phi^{-1}((2k-1)/16)) la migracion
     maxima en octiles sobre una rejilla fina de sorpresas s en [-10, 10];
  3. la migracion maxima desde CUALQUIER latente z en [-3, 3] (rejilla fina),
     para acotar el numero de fronteras que una unica sorpresa puede cruzar.

Uso: python scripts/verificar_paso_maximo.py
Salidas: results/verificacion_paso_maximo_octiles.csv y
         results/verificacion_paso_maximo_perfiles.csv
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np
import pandas as pd

from motor_owa.adaptive import update_latent
from motor_owa.config import EngineConfig, PROFILE_NAMES
from motor_owa.latent import classify_z, octile_z, phi_inv

RES = os.path.join(HERE, "..", "results")
cfg = EngineConfig()
kappa, lam = cfg.kappa, cfg.loss_lambda

cortes = np.array([phi_inv(j / 8) for j in range(1, 8)])
anchos = np.diff(cortes)                       # 6 octiles interiores (2..7)
pares = anchos[:-1] + anchos[1:]               # 5 pares contiguos interiores
filas = []
for i, a in enumerate(anchos):
    filas.append({"tipo": "octil", "octiles": f"{i + 2}",
                  "desde_z": cortes[i], "hasta_z": cortes[i + 1], "ancho": a})
for i, a in enumerate(pares):
    filas.append({"tipo": "par", "octiles": f"{i + 2}-{i + 3}",
                  "desde_z": cortes[i], "hasta_z": cortes[i + 2], "ancho": a})
filas.append({"tipo": "paso_max_prudencia", "octiles": "",
              "desde_z": np.nan, "hasta_z": np.nan, "ancho": kappa * lam})
filas.append({"tipo": "paso_max_audacia", "octiles": "",
              "desde_z": np.nan, "hasta_z": np.nan, "ancho": kappa})
pd.DataFrame(filas).to_csv(
    os.path.join(RES, "verificacion_paso_maximo_octiles.csv"), index=False)

s_grid = np.linspace(-10, 10, 4001)
perf = []
for k in range(1, 9):
    z0 = octile_z(k)
    mig = np.array([classify_z(update_latent(z0, s, kappa, lam)) - k
                    for s in s_grid])
    perf.append({"perfil": PROFILE_NAMES[k - 1], "z_ancla": z0,
                 "mig_max_prudencia": int(mig.min()),
                 "mig_max_audacia": int(mig.max())})
# cualquier latente de partida
z_grid = np.linspace(-3, 3, 6001)
worst_down = min(classify_z(update_latent(z, -50.0, kappa, lam)) - classify_z(z)
                 for z in z_grid)
worst_up = max(classify_z(update_latent(z, 50.0, kappa, lam)) - classify_z(z)
               for z in z_grid)
perf.append({"perfil": "cualquier z en [-3, 3]", "z_ancla": np.nan,
             "mig_max_prudencia": int(worst_down),
             "mig_max_audacia": int(worst_up)})
out = pd.DataFrame(perf)
out.to_csv(os.path.join(RES, "verificacion_paso_maximo_perfiles.csv"),
           index=False)
print(pd.DataFrame(filas).round(4).to_string())
print(out.round(4).to_string())
