# -*- coding: utf-8 -*-
"""2-maqola, Figure 3: o'lchangan federativ yangilanish spektri va rang qonuni.

(a) Amazon Reviews 2023 oqimida o'qitilgan BPR-MF ning silo-darajali
    gradient matritsasi singulyar qiymatlari; power-law va eksponensial
    moslashtirish solishtiriladi.
(b) O'lchangan optimal uzatish rangi r*(eps); qiyalik nazariy 2/(zeta+1)
    bilan solishtiriladi.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
N = os.path.join(ROOT, "natijalar")
Z = json.load(open(os.path.join(N, "zeta_rekonstruksiya_zich.json")))
sp = json.load(open(os.path.join(N, "spektr_toliq.json")))["oxirgi_raund"]

C1, C2, C3 = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D9D9D9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})

fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                           gridspec_kw={"wspace": 0.30})

# --- (a) spektr ------------------------------------------------------
sv = np.array(sp["sv_barchasi"], float)
k = np.arange(1, len(sv) + 1)
z = sp["power_law_zeta"]
a.loglog(k, sv, color=C1, lw=0, marker="o", ms=3.2, label="measured spectrum")
a.loglog(k, np.exp(sp["power_law_c"]) * k ** sp["power_law_qiyalik"],
         color=C2, lw=1.4, ls="--",
         label=fr"power law $\zeta={z:.3f}$,  $R^2={sp['power_law_R2']:.3f}$")
a.loglog(k, np.exp(sp["exp_c"] + sp["exp_qiyalik"] * k), color=C3, lw=1.4,
         ls=":", label=fr"exponential,  $R^2={sp['eksponensial_R2']:.3f}$")
a.set_xlabel("Singular-value index $i$")
a.set_ylabel(r"$\sigma_i$")
a.legend(frameon=False, fontsize=7.5, loc="lower left")
a.set_title("(a) measured update spectrum", fontsize=9, color=INK, pad=10)

# --- (b) rang qonuni -------------------------------------------------
eps = np.array(Z["eps_grid"], float)
r = np.array(Z["r_yulduz"], float)
d2 = sp["n_sv"]
ichki = (r > 1) & (r < d2)
a_slope = Z["olchangan_qiyalik"]
b_pred = Z["kutilgan_korsatkich"]
b.loglog(eps[ichki], r[ichki], "o", color=C1, ms=5, label="interior optima")
b.loglog(eps[~ichki], r[~ichki], "o", mfc="none", mec=MUTED, ms=5,
         label="boundary-clipped (excluded)")
xx = np.logspace(np.log10(eps[ichki].min()), np.log10(eps[ichki].max()), 50)
c0 = np.exp(np.mean(np.log(r[ichki]) - a_slope * np.log(eps[ichki])))
b.loglog(xx, c0 * xx ** a_slope, color=C1, lw=1.4,
         label=fr"fit: slope ${a_slope:.3f}$ ($R^2={Z['qiyalik_R2']:.3f}$)")
c1 = np.exp(np.mean(np.log(r[ichki]) - b_pred * np.log(eps[ichki])))
b.loglog(xx, c1 * xx ** b_pred, color=C2, lw=1.3, ls="--",
         label=fr"prediction $2/(\zeta+1)={b_pred:.3f}$")
b.axhline(d2, color=MUTED, lw=0.8, ls=":")
b.text(eps.max() * 0.98, d2 * 0.72, r"rank ceiling $r=d_2$", fontsize=7.5,
       color=MUTED, ha="right", va="top")
b.set_xlabel(r"Privacy budget $\varepsilon$")
b.set_ylabel(r"Optimal transmitted rank $r^*$")
b.legend(frameon=False, fontsize=7.2, loc="upper left", handlelength=1.6, borderaxespad=0.2)
b.set_title("(b) rank law on the measured stream", fontsize=9, color=INK, pad=10)

b.set_ylim(0.7, d2*4.5)
out = os.path.join(ROOT, "figures"); os.makedirs(out, exist_ok=True)
plt.savefig(os.path.join(out, "a2_fig3.png")); plt.close()
print("a2_fig3.png saqlandi; qiyalik", a_slope, "kutilgan", b_pred,
      "ichki nuqta", int(ichki.sum()))
