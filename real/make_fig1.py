# -*- coding: utf-8 -*-
"""2-maqola, Figure 1: sindirish (break-even) egri chizigi + DP amallar soni.

Chap panel: qat'iy lokalizatsiya komplayens narxi tashqi/mahalliy vCPU narx
nisbatiga qarab. O'lchangan nisbat (3,84) egri chiziqdan ancha o'ngda.
O'ng panel: aniq qayta rejalashtirish amallari N|L|^2.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "natijalar", "joylashtirish.json")))
sw = R["sindirish_sweep"]
nisbat = np.array([s["L2_L1_nisbat"] for s in sw])
prem = np.array([s["komplayens_foiz"] for s in sw])
haqiqiy = R["haqiqiy_L2_L1_nisbat"]

C1, C2, C3 = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D9D9D9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})

fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9),
                           gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.32})

o = np.argsort(nisbat)
a.plot(nisbat[o], prem[o], color=C2, lw=1.6, marker="o", ms=3.2,
       label="strict localization vs unconstrained")
# nol mintaqa
a.axhline(0, color=MUTED, lw=0.8, ls=":")
a.axvspan(0.47, 4.6, color=C3, alpha=0.10)
a.text(1.35, 200, "zero compliance cost", fontsize=8, color=C3, ha="center")
# sindirish nuqtasi
a.axvline(0.47, color=C1, lw=1.0, ls="--")
a.annotate("break-even\n$L_2/L_1 \\approx 0.47$", xy=(0.47, 60), xytext=(0.115, 430),
           fontsize=8, color=C1,
           arrowprops=dict(arrowstyle="->", color=C1, lw=0.8))
# o'lchangan nisbat
a.scatter([haqiqiy], [0], color=C1, s=30, zorder=6, clip_on=False)
a.annotate(f"measured: {haqiqiy:.2f}\n(AWS Frankfurt / UzCloud)",
           xy=(haqiqiy, 0), xytext=(0.62, 640), fontsize=8, color=MUTED,
           arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
a.set_xscale("log")
a.set_xlim(0.045, 4.6)
a.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
a.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "4"])
a.set_xlabel("External / domestic vCPU-hour price ratio  $L_2/L_1$")
a.set_ylabel("Compliance premium (%)")
a.set_ylim(-40, 920)

Ns = np.arange(2, 41)
b.plot(Ns, Ns * 9, color=C1, lw=1.5, label=r"$|\mathcal{L}|=3$")
b.plot(Ns, Ns * 25, color=C2, lw=1.5, ls="--", label=r"$|\mathcal{L}|=5$")
b.scatter([8], [72], color=C1, s=25, zorder=5)
b.annotate("this study: 72 ops", xy=(8, 72), xytext=(14, 260), fontsize=8,
           color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
b.set_xlabel("Pipeline stages $N$")
b.set_ylabel(r"DP operations $N|\mathcal{L}|^2$")
b.legend(frameon=False, fontsize=8)

out = os.path.join(ROOT, "figures"); os.makedirs(out, exist_ok=True)
plt.savefig(os.path.join(out, "a2_fig1.png")); plt.close()
print("a2_fig1.png saqlandi. break-even =", 
      [s['L2_L1_nisbat'] for s in sw if s['komplayens_foiz'] > 0][0])
