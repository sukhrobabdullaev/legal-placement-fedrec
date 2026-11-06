# -*- coding: utf-8 -*-
"""Figures 2 and 3 (revised).
Figure 2: (a) singular-value spectra of the clipped aggregate at the last
recorded round for each corpus (normalised by s_1); (b) the exact optimal
rank r*(sigma) of Proposition 1 for the same spectra.
Figure 3: Recall@20 of private federated BPR-MF on MovieLens-1M against the
transmitted rank, one line per total budget, with the popularity and
non-private references.
Run from the package root after v2_spectra.py and v2_downstream_dp.py."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from adaptrec import dp

INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D9D9D9"
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.dpi": 220,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})
V2 = os.path.join(ROOT, "natijalar", "v2")
FIG = os.path.join(ROOT, "figures")

# ---------------- Figure 2 -------------------------------------------------
corp = [("ml-1m", "MovieLens-1M"), ("gowalla", "Gowalla"), ("yelp2018", "Yelp2018"),
        ("amazon-book", "Amazon-Book")]
old = json.load(open(os.path.join(ROOT, "real", "natijalar", "spektr_toliq.json")))["oxirgi_raund"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.32})
sig = np.geomspace(1e-3, 1e3, 241)
k = np.arange(1, 65)
for (c, lab), col in zip(corp, PAL):
    o = json.load(open(os.path.join(V2, "spectra_%s.json" % c)))
    p = o["by_round"][str(o["rounds"])]
    s = np.array(p["sv"])
    a.plot(k, s / s[0], color=col, lw=1.3, label=r"%s ($\zeta$ = %.2f)" % (lab, p["zeta_all"]))
    rs = [dp.optimal_rank_threshold(s ** 2, o["n_item"], x) for x in sig]
    b.plot(1 / sig * np.sqrt(o["n_item"]) / s[0], rs, color=col, lw=1.3, label=lab)
s = np.array(old["sv_barchasi"])
a.plot(k, s / s[0], color=PAL[4], lw=1.3, ls="--",
       label=r"Video Games, v1 ($\zeta$ = %.2f)" % old["power_law_zeta"])
rs = [dp.optimal_rank_threshold(s ** 2, 2048, x) for x in sig]
b.plot(1 / sig * np.sqrt(2048) / s[0], rs, color=PAL[4], lw=1.3, ls="--", label="Video Games, original")
for zeta, ls in ((1.0, ":"),):
    ref = k ** (-(zeta + 1) / 2)
    a.plot(k, ref, color=MUTED, lw=0.9, ls=ls, label=r"power law, $\zeta$ = 1")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"Singular-value index $j$"); a.set_ylabel(r"$s_j / s_1$")
a.legend(frameon=False, fontsize=6.3, loc="lower left")
a.set_title("(a) Spectra of the clipped aggregate", fontsize=8.5, loc="left")
b.set_xscale("log"); b.set_yscale("log")
b.set_xlim(0.3, 20); b.set_ylim(0.8, 80)
b.set_xlabel(r"Signal-to-noise  $s_1 / (\sqrt{d_1}\,\sigma)$")
b.set_ylabel(r"Optimal rank $r^*$ (Proposition 1)")
b.set_title("(b) Exact optimal rank", fontsize=8.5, loc="left")
plt.savefig(os.path.join(FIG, "v2_fig2_spectra.png")); plt.close()

# ---------------- Figure 3 -------------------------------------------------
D = json.load(open(os.path.join(V2, "downstream_ml-1m.json")))
eps_list = sorted({g["eps_total"] for g in D["grid"]})
ranks = sorted({g["rank"] for g in D["grid"]})
fig, ax = plt.subplots(figsize=(5.4, 3.0))
for e, col in zip(eps_list, PAL):
    m = [np.mean([g["recall20"] for g in D["grid"] if g["eps_total"] == e and g["rank"] == r])
         for r in ranks]
    sd = [np.std([g["recall20"] for g in D["grid"] if g["eps_total"] == e and g["rank"] == r], ddof=1)
          for r in ranks]
    ax.errorbar(ranks, m, yerr=sd, color=col, lw=1.3, marker="o", ms=3, capsize=1.5,
                label=r"$\varepsilon_{\mathrm{tot}}$ = %g" % e)
    i = int(np.argmax(m))
    ax.scatter([ranks[i]], [m[i]], s=60, facecolors="none", edgecolors=col, lw=1.2, zorder=6)
npv = np.mean([x["recall20"] for x in D["nonprivate"]["full_rank"]])
ax.axhline(npv, color=INK, lw=0.8, ls="--")
ax.text(66, npv, "non-private,\nfull rank", fontsize=6.8, color=INK, va="center")
ax.axhline(D["popularity"]["recall20"], color=MUTED, lw=0.8, ls=":")
ax.text(66, D["popularity"]["recall20"], "popularity", fontsize=6.8, color=MUTED, va="center")
ax.set_xscale("log", base=2); ax.set_xticks(ranks); ax.set_xticklabels(ranks)
ax.set_xlabel(r"Transmitted rank $r$"); ax.set_ylabel("Recall@20 (MovieLens-1M)")
ax.legend(frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(1.17, 1.0))
plt.savefig(os.path.join(FIG, "v2_fig3_downstream.png")); plt.close()
print("saved v2_fig2_spectra.png, v2_fig3_downstream.png")
