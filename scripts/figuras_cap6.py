"""Figuras 6.1-6.3 de la monografia sobre datos reales de EE. UU. versionados.

Lee el snapshot versionado del OE4 (../validacion-oe4/data/snapshot_oe4/
us_precios.csv), verifica su SHA-256 contra MANIFEST.json y genera, con
rotulos en espanol y coma decimal:

  figures/es/fig6_1_anclas_orness.png      escalera de anclas (octiles)
  figures/es/fig6_2_carteras_us.png        sigma objetivo vs alcanzada (ex ante)
                                           y riesgo-retorno esperado, ultima
                                           fecha con horizonte realizado
  figures/es/fig6_3_trayectoria_us.png     trayectoria adaptativa (canal
                                           automatico) de un inversor Pragmatico
y los datos subyacentes en results/us/.

Uso: python scripts/figuras_cap6.py [--snapshot DIR]
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from motor_owa.adaptive import InvestorState
from motor_owa.config import EngineConfig, PROFILE_NAMES
from motor_owa.data import load_csv
from motor_owa.engine import RecommendationEngine
from motor_owa.latent import phi_inv
from motor_owa.profiles import all_profiles
from motor_owa.viz_es import (NOMBRES_ES, OKABE_ITO, coma, ejes_coma, estilo,
                              guardar)

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", default=os.path.join(
    HERE, "..", "..", "validacion-oe4", "data", "snapshot_oe4"))
args = ap.parse_args()
csv = os.path.join(args.snapshot, "us_precios.csv")
man = json.load(open(os.path.join(args.snapshot, "MANIFEST.json")))
sha = hashlib.sha256(open(csv, "rb").read()).hexdigest()
assert sha == man["archivos"]["us_precios.csv"], "SHA-256 no coincide"

RES = os.path.join(HERE, "..", "results", "us")
FIG = os.path.join(HERE, "..", "figures", "es")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

px = load_csv(csv)
cfg = EngineConfig()
eng = RecommendationEngine(px, cfg)
profs = all_profiles(cfg.anchors)
es = [NOMBRES_ES[p.name] for p in profs]

# ---------------- Figura 6.1: anclas ----------------
fig, ax = plt.subplots(figsize=(8, 4.3))
alphas = [p.alpha for p in profs]
ax.bar(es, alphas, color=OKABE_ITO, edgecolor="black", linewidth=0.6)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1,
           label="neutralidad (orness = 0,5)")
for i, a in enumerate(alphas):
    ax.text(i, a + 0.02, coma(a, 4), ha="center", fontsize=8.5)
ax.set_ylabel("Orness del perfil  $\\alpha_k = (2k-1)/16$")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=9, loc="upper left")
estilo(ax); ejes_coma(ax, x=False)
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
guardar(fig, os.path.join(FIG, "fig6_1_anclas_orness.png"))

# ---------------- Figura 6.2: carteras en la ultima fecha evaluable ----------
t = len(px) - cfg.horizon - 1
ports = eng.builder.build_all(profs, t)
rows = []
for p in profs:
    r = ports[p.name]
    seg = px[r.weights.index].iloc[t:t + cfg.horizon + 1].pct_change().dropna()
    vol_real = float((seg @ r.weights.values).std() * np.sqrt(252))
    rows.append({"perfil": p.name, "perfil_es": NOMBRES_ES[p.name],
                 "orness": r.alpha, "vol_objetivo": r.target_vol,
                 "vol_alcanzada_ex_ante": r.expected_vol,
                 "vol_realizada_63d": vol_real,
                 "retorno_esperado_ex_ante": r.expected_return,
                 "n_activos": len(r.selected), "activos": " ".join(r.selected)})
car = pd.DataFrame(rows)
car.insert(0, "fecha", str(px.index[t].date()))
car.to_csv(os.path.join(RES, "carteras_por_perfil.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax = axes[0]
lo = min(car.vol_objetivo.min(), car.vol_alcanzada_ex_ante.min()) * 100 - 1
hi = max(car.vol_objetivo.max(), car.vol_alcanzada_ex_ante.max()) * 100 + 1
ax.plot([lo, hi], [lo, hi], color="gray", linestyle=":", linewidth=1,
        label="alcanzada = objetivo")
for i, r in car.iterrows():
    ax.scatter(r.vol_objetivo * 100, r.vol_alcanzada_ex_ante * 100, s=80,
               color=OKABE_ITO[i], edgecolor="black", linewidth=0.6, zorder=3,
               label=r.perfil_es)
ax.set_xlabel("Volatilidad objetivo $\\sigma^*_k$ (% anual)")
ax.set_ylabel("Volatilidad alcanzada ex ante (% anual)")
ax.set_title("(a) Objetivo frente a alcanzada", fontsize=10.5)
estilo(ax); ejes_coma(ax)
ax = axes[1]
ax.plot(car.vol_alcanzada_ex_ante * 100, car.retorno_esperado_ex_ante * 100,
        color="#BBBBBB", linewidth=1.2, zorder=1)
for i, r in car.iterrows():
    ax.scatter(r.vol_alcanzada_ex_ante * 100, r.retorno_esperado_ex_ante * 100,
               s=80, color=OKABE_ITO[i], edgecolor="black", linewidth=0.6,
               zorder=3)
ax.set_xlabel("Volatilidad alcanzada ex ante (% anual)")
ax.set_ylabel("Retorno esperado ex ante (% anual)")
ax.set_title("(b) Riesgo y retorno esperados", fontsize=10.5)
estilo(ax); ejes_coma(ax)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=9, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.07))
fig.tight_layout(rect=[0, 0.04, 1, 1])
guardar(fig, os.path.join(FIG, "fig6_2_carteras_us.png"))

# ---------------- Figura 6.3: trayectoria adaptativa ----------------
st = InvestorState.from_profile("Pragmatist", cfg.anchors)
t0 = cfg.lookback
recs = eng.simulate_investor(st, t0, n_cycles=12)
hist = pd.DataFrame(st.history)
hist.insert(0, "fecha_inicio", [str(px.index[t0 + i * cfg.horizon].date())
                                for i in range(len(hist))])
hist["perfil_antes"] = [NOMBRES_ES[PROFILE_NAMES[k - 1]] for k in hist.k_before]
hist["perfil_despues"] = [NOMBRES_ES[PROFILE_NAMES[k - 1]] for k in hist.k_after]
hist.to_csv(os.path.join(RES, "trayectoria_pragmatico.csv"), index=False)

fig, ax = plt.subplots(figsize=(9, 4.6))
z = list(hist.z_before) + [hist.z_after.iloc[-1]]
fechas = list(hist.fecha_inicio) + [str(px.index[min(t0 + len(hist) * cfg.horizon,
                                                     len(px) - 1)].date())]
ax.plot(range(len(z)), z, marker="o", color=OKABE_ITO[5], linewidth=1.6,
        zorder=3, label="latente $z_t$ del inversor")
bordes = [phi_inv(j / 8) for j in range(1, 8)]
for zb in bordes:
    ax.axhline(zb, color="gray", linestyle=":", linewidth=0.8)
centros = [-2.4] + [(bordes[j] + bordes[j + 1]) / 2 for j in range(6)] + [1.9]
for k, zc in enumerate(centros):
    ax.text(len(z) - 0.55, zc, NOMBRES_ES[PROFILE_NAMES[k]], fontsize=7.5,
            va="center", color="#444444")
primera = True
for i, r in hist.iterrows():
    if r.migrated:
        ax.axvline(i + 1, color=OKABE_ITO[6], alpha=0.55, linewidth=1.2,
                   label="migración de perfil" if primera else None)
        primera = False
ax.set_xticks(range(len(z)))
ax.set_xticklabels([f[:7] for f in fechas], rotation=35, ha="right", fontsize=8)
ax.set_xlabel("Cierre de horizonte (año-mes)")
ax.set_ylabel("Apetito de riesgo latente $z_t$")
ax.set_ylim(-3.2, 2.3)
ax.set_xlim(-0.3, len(z) + 0.9)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
estilo(ax); ejes_coma(ax, x=False)
guardar(fig, os.path.join(FIG, "fig6_3_trayectoria_us.png"))

print(car[["perfil_es", "orness", "vol_objetivo", "vol_alcanzada_ex_ante",
           "vol_realizada_63d", "retorno_esperado_ex_ante"]].round(4).to_string())
print(f"fecha de la cartera: {px.index[t].date()}")
print(hist[["fecha_inicio", "surprise", "z_before", "z_after", "perfil_antes",
            "perfil_despues", "migrated"]].round(3).to_string())
