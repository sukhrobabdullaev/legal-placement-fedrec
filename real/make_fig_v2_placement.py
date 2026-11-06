# -*- coding: utf-8 -*-
"""Figure 1 (revised): (a) compliance premium of the 2021 broad-localization
mask over the unrestricted plan on the (L2/L1, L0/L1) price plane, with the
observed external providers marked; (b) cost per million requests by provider
under the 2021 regime and the 2026 regime with an EU (adequacy-listed)
destination. Run from real/ after run_joylashtirish_v2.py."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D9D9D9"
C1, C2, C3 = "#0072B2", "#D55E00", "#009E73"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": False, "figure.dpi": 220,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})

o = json.load(open(os.path.join(HERE, "natijalar", "v2", "placement_v2.json")))
m = o["breakeven_map"]
xs, ys = np.array(m["L2_over_L1"]), np.array(m["L0_over_L1"])
Z = np.array(m["premium_pct"]["egress_0.0011"]); Z9 = np.array(m["premium_pct"]["egress_0.09"])

fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.75})
Zp = np.where(Z > 0.05, Z, np.nan)
pc = a.pcolormesh(xs, ys, Zp, shading="nearest", cmap="Oranges",
                  norm=LogNorm(vmin=1, vmax=np.nanmax(Zp)))
a.contour(xs, ys, Z, levels=[0.05], colors=[INK], linewidths=0.9)
a.contour(xs, ys, Z9, levels=[0.05], colors=[INK], linewidths=0.8, linestyles="--")

a.set_xscale("log"); a.set_yscale("log")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
fmt = FuncFormatter(lambda v, _: ("%g" % v))
a.xaxis.set_major_locator(FixedLocator([0.1, 0.25, 0.5, 1, 2, 4])); a.xaxis.set_major_formatter(fmt)
a.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2])); a.yaxis.set_major_formatter(fmt)
a.xaxis.set_minor_locator(NullLocator()); a.yaxis.set_minor_locator(NullLocator())
a.set_xlabel("External / domestic-cloud vCPU-hour price  $L_2/L_1$")
a.set_ylabel("On-premises / domestic-cloud price  $L_0/L_1$")
cb = fig.colorbar(pc, ax=a, pad=0.03, fraction=0.05); cb.set_label("Premium of 2021 mask (%)", fontsize=7.5)
cb.ax.yaxis.set_major_formatter(fmt)
head = [s for s in o["scenarios"] if s["profile"] == "measured (x86-64)"
        and s["L0"] == "on-prem, UZ colocation" and s["L1"] == "UzCloud, 4 GB/vCPU"
        and s["uplink_util"] == 0.3]
for s in head:
    a.scatter([s["ratio_L2_L1"]], [s["ratio_L0_L1"]], s=16, color=C1, zorder=5,
              edgecolor="white", linewidth=0.5)
a.annotate("Hetzner CX23", (head[-1]["ratio_L2_L1"], 0.491), xytext=(0.12, 0.33),
           fontsize=7, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
a.annotate("AWS, GCP,\nHetzner CCX13", (2.4, 0.491), xytext=(1.25, 0.8),
           fontsize=7, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
a.axhline(0.491, color=MUTED, lw=0.5, ls=":")
a.set_title("(a) Where localization costs money", fontsize=8.5, loc="left")

names = [s["L2"] for s in head]
short = [n.replace(" m7i.large", "").replace(" e2-standard-2", "").replace(" (dedicated)", "")
         .replace(" (shared)", "") for n in names]
r21 = [s["regimes"]["B. 2021 Art. 27-1 (broad reading)"]["usd_1M"] for s in head]
r26 = [s["regimes"]["C1. 2026, EU destination on adequacy list"]["usd_1M"] for s in head]
y = np.arange(len(names))
b.scatter(r21, y, s=34, color=C2, zorder=4, label="2021 broad localization")
b.scatter(r26, y, s=18, color=C3, zorder=5, label="2026, EU destination listed")
for yi, (u, v) in enumerate(zip(r21, r26)):
    if abs(u - v) > 1e-6:
        b.annotate("", xy=(v, yi), xytext=(u, yi),
                   arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
        b.text((u + v) / 2, yi - 0.35, "-%.0f%%" % ((1 - v / u) * 100), ha="center",
               fontsize=7, color=MUTED)
b.set_yticks(y); b.set_yticklabels(short, fontsize=7.5); b.invert_yaxis()
b.set_xlim(0.15, 0.29); b.grid(axis="x", color=GRID, lw=0.6)
b.set_xlabel("Optimal cost (USD per million requests)")
b.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1)
b.set_title("(b) Cost by external provider", fontsize=8.5, loc="left")
out = os.path.join(os.path.dirname(HERE), "figures"); os.makedirs(out, exist_ok=True)
plt.savefig(os.path.join(out, "v2_fig1_placement.png")); plt.close()
print("saved v2_fig1_placement.png")
