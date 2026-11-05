# -*- coding: utf-8 -*-
"""2-maqola, 6.1-bo'lim: uchta huquqiy rejim ssenariysida joylashtirish.

Sakkiz bosqichli qayta ishlash oqimi uchta qatlam (o'z HM / mahalliy bulut /
tashqi bulut) ustida joylashtiriladi. Har ssenariy uchun optimal reja va
narx dinamik dasturlash bilan (72 amal) hisoblanadi, so'ng Figure 1 chiziladi.

Ishga tushirish (paket ildizidan):  python experiments/placement_scenarios.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from adaptrec.placement import joylashtirish_dp

C1, C2, C3 = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D9D9D9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})

N, L = 8, 3
BOSQICHLAR = ["ingest", "feature", "retrieval", "pre-rank", "ranking",
              "re-rank", "logging", "analytics"]
QATLAMLAR = ["on-prem DC", "domestic cloud", "external cloud"]

# Bajarish narxlari (bosqich x qatlam): tashqi bulut birligi eng arzon
base_cost = np.array([
    [3.0, 2.0, 1.2], [3.2, 2.2, 1.3], [2.8, 2.0, 1.5], [2.4, 1.8, 1.2],
    [2.6, 1.9, 1.1], [2.2, 1.6, 1.0], [1.6, 1.2, 0.8], [2.0, 1.4, 0.9]])
# Qatlamlararo uzatish narxi (egress + shifrlash)
move = np.array([[0.0, 0.6, 1.4], [0.6, 0.0, 0.9], [1.4, 0.9, 0.0]])

SSENARIYLAR = {
    "No restrictions": np.ones((N, L), bool),
    "Strict localization\n(PD stages on-prem)": np.array(
        [[1,0,0],[1,1,0],[1,1,1],[1,1,1],[1,1,0],[1,1,1],[1,0,0],[1,1,0]], bool),
    "Conditional regime\n(2026 amendments)": np.array(
        [[1,1,0],[1,1,0],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,0],[1,1,1]], bool),
}

labels, costs, plans = [], [], []
for nom, mask in SSENARIYLAR.items():
    reja, narx = joylashtirish_dp(base_cost, move, mask)
    labels.append(nom); costs.append(narx); plans.append(reja)
    print(f"{nom.replace(chr(10), ' ')}: narx={narx:.1f}, "
          f"reja={[QATLAMLAR[l] for l in reja]}")

fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.8),
                           gridspec_kw={"width_ratios": [1.15, 1]})
a.barh(range(len(labels))[::-1], costs, color=[C3, C2, C1], height=0.55)
a.set_yticks(range(len(labels))[::-1]); a.set_yticklabels(labels, fontsize=8)
a.set_xlabel("Optimal plan cost (arb. units)")
for i, c in enumerate(costs):
    a.text(c + 0.15, len(labels) - 1 - i, f"{c:.1f}", va="center", fontsize=8)
Ns = np.arange(2, 41)
b.plot(Ns, Ns * 9, color=C1, lw=1.5, label=r"$|\mathcal{L}|=3$")
b.plot(Ns, Ns * 25, color=C2, lw=1.5, ls="--", label=r"$|\mathcal{L}|=5$")
b.scatter([8], [72], color=C1, s=25, zorder=5)
b.annotate("this study: 72 ops", xy=(8, 72), xytext=(14, 260), fontsize=8,
           color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
b.set_xlabel("Pipeline stages $N$"); b.set_ylabel(r"DP operations $N|\mathcal{L}|^2$")
b.legend(frameon=False, fontsize=8)
out = os.path.join(ROOT, "figures"); os.makedirs(out, exist_ok=True)
plt.savefig(os.path.join(out, "a2_fig1_sintetik.png")); plt.close()

os.makedirs(os.path.join(ROOT, "natijalar"), exist_ok=True)
json.dump({nom.replace("\n", " "): {"cost": round(c, 2), "plan": p}
           for nom, c, p in zip(SSENARIYLAR, costs, plans)},
          open(os.path.join(ROOT, "natijalar", "placement_results_sintetik.json"), "w"),
          indent=1)
print("a2_fig1_sintetik.png saqlandi (maqoladagi Figure 1 esa\n       real/make_fig1.py dan chiqadi: real narxlardagi sindirish egri chizigi).")
