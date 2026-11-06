# -*- coding: utf-8 -*-
"""Publication figures for the revised article (Figures 1-3).

Full MDPI text width (174 mm), Liberation Sans (Arial metrics), 600 dpi PNG
for the manuscript plus vector PDF for the journal. Colours: categorical
slots in fixed order for identities, one blue ramp (validated ordinal) for
the ordered privacy budgets, one orange ramp for the premium magnitude.
Every figure is checked for overlapping text boxes before it is saved.

Run from the package root after the v2 experiments.
"""
import json, os, sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from adaptrec import dp

V2 = os.path.join(ROOT, "natijalar", "v2")
FIG = os.path.join(ROOT, "figures")
MM = 1 / 25.4
W_FULL = 174 * MM

# ---- palette (validated with the dataviz validator) ----------------------
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0b2a52"]   # ordinal, light->dark
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ORANGES = LinearSegmentedColormap.from_list(
    "or", ["#fde3d3", "#f7b48d", "#eb6834", "#c4481a", "#7f2a0a"])

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 6, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8, "axes.grid": False, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7.5,
    "savefig.dpi": 600, "figure.dpi": 150, "pdf.fonttype": 42, "svg.fonttype": "none"})

plain = FuncFormatter(lambda v, _: "%g" % v)


def check_overlaps(fig, name):
    """Fail loudly if any two visible text boxes overlap."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.width > 0 and bb.height > 0:
            boxes.append((t.get_text()[:30], bb))
    bad = []
    for (a, ba), (b, bb) in itertools.combinations(boxes, 2):
        if ba.overlaps(bb):
            ix = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            iy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if ix > 1 and iy > 1:
                bad.append((a, b))
    if bad:
        raise SystemExit("%s: overlapping text %s" % (name, bad))
    print("%s: %d text boxes, no overlaps" % (name, len(boxes)))


def save(fig, stem):
    check_overlaps(fig, stem)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (stem, ext)), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def spread(ys, gap):
    """Move label positions (already sorted by value) apart by at least `gap`."""
    ys = list(ys)
    for _ in range(200):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                d = (gap - (ys[i] - ys[i - 1])) / 2
                ys[i - 1] -= d; ys[i] += d; moved = True
        if not moved:
            break
    return ys


# =========================================================================
# Figure 1 - placement
# =========================================================================
def figure1():
    o = json.load(open(os.path.join(ROOT, "real", "natijalar", "v2", "placement_v2.json")))
    m = o["breakeven_map"]
    xs, ys = np.array(m["L2_over_L1"]), np.array(m["L0_over_L1"])
    Z = np.array(m["premium_pct"]["egress_0.0011"])
    Z9 = np.array(m["premium_pct"]["egress_0.09"])
    head = [s for s in o["scenarios"] if s["profile"] == "measured (x86-64)"
            and s["L0"] == "on-prem, UZ colocation" and s["L1"] == "UzCloud, 4 GB/vCPU"
            and s["uplink_util"] == 0.3]
    l0r = head[0]["ratio_L0_L1"]

    fig = plt.figure(figsize=(W_FULL, 72 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.045, 0.95], wspace=0.12,
                          left=0.07, right=0.985, bottom=0.17, top=0.86)
    a = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1])
    b = fig.add_subplot(gs[0, 2])
    b_pos = b.get_position(); b.set_position([b_pos.x0 + 0.2, b_pos.y0, b_pos.width - 0.2, b_pos.height])

    # (a) premium map
    Zp = np.where(Z > 0.05, Z, np.nan)
    pc = a.pcolormesh(xs, ys, Zp, shading="nearest", cmap=ORANGES,
                      norm=LogNorm(vmin=1, vmax=400), rasterized=True)
    a.contour(xs, ys, Z, levels=[0.05], colors=[INK], linewidths=1.0)
    a.contour(xs, ys, Z9, levels=[0.05], colors=[INK], linewidths=0.9, linestyles=[(0, (3.5, 2))])
    a.set_xscale("log"); a.set_yscale("log")
    a.xaxis.set_major_locator(FixedLocator([0.1, 0.25, 0.5, 1, 2, 4])); a.xaxis.set_major_formatter(plain)
    a.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2])); a.yaxis.set_major_formatter(plain)
    for ax_ in (a.xaxis, a.yaxis):
        ax_.set_minor_locator(NullLocator())
    a.set_xlim(xs.min(), xs.max()); a.set_ylim(ys.min(), ys.max())
    a.set_xlabel(r"External / domestic-cloud price  $L_2/L_1$")
    a.set_ylabel(r"On-premises / domestic-cloud price  $L_0/L_1$")
    a.set_title("a  Premium of the 2021 rule")
    a.axhline(l0r, color=INK2, lw=0.6, ls=(0, (1, 1.5)))
    pts = sorted(s["ratio_L2_L1"] for s in head)
    a.scatter(pts, [l0r] * len(pts), s=22, color="white", edgecolor=INK, linewidth=0.9, zorder=6)
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9)
    a.annotate("Hetzner CX23", xy=(pts[0], l0r), xytext=(0.15, 0.36), fontsize=7, color=INK,
               ha="center", va="center", bbox=box,
               arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=0, shrinkB=3))
    a.annotate("", xy=(2.6, l0r * 1.07), xytext=(2.6, l0r * 1.55),
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    a.text(2.6, l0r * 1.6, "AWS, Google Cloud,\nHetzner CCX13", fontsize=7, color=INK,
           ha="center", va="bottom", linespacing=1.15)
    a.text(3.9, l0r * 0.93, "assumed on-premises price", fontsize=6.5, color=INK2,
           ha="right", va="top")
    a.text(2.0, 1.75, "no premium", fontsize=7.5, color=INK2, ha="center", va="center", style="italic")
    leg = [Line2D([], [], color=INK, lw=1.0, label="egress 0.001 USD/GB"),
           Line2D([], [], color=INK, lw=0.9, ls=(0, (3.5, 2)), label="egress 0.09 USD/GB")]
    a.legend(handles=leg, loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=6.8,
             handlelength=2.2, borderaxespad=0.4, title="Break-even boundary", title_fontsize=6.8,
             alignment="left")
    cb = fig.colorbar(pc, cax=cax)
    cb.set_ticks([1, 10, 100]); cb.ax.yaxis.set_major_formatter(plain)
    cb.ax.yaxis.set_minor_locator(NullLocator())
    cb.set_label("Premium (%)", fontsize=7.5, labelpad=2)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor(AXIS)

    # (b) cost by offer, 2021 vs 2026 (EU destination listed)
    lab = {"AWS m7i.large on-demand": "AWS on-demand", "AWS m7i.large RI 1-yr": "AWS 1-yr reserved",
           "AWS m7i.large RI 3-yr": "AWS 3-yr reserved", "AWS m7i.large spot": "AWS spot",
           "GCP e2-standard-2 on-demand": "Google Cloud", "Hetzner CCX13 (dedicated)": "Hetzner CCX13",
           "Hetzner CX23 (shared)": "Hetzner CX23"}
    rows = sorted(head, key=lambda s: -s["ratio_L2_L1"])
    y = np.arange(len(rows))
    c21 = [s["regimes"]["B. 2021 Art. 27-1 (broad reading)"]["usd_1M"] for s in rows]
    c26 = [s["regimes"]["C1. 2026, EU destination on adequacy list"]["usd_1M"] for s in rows]
    b.set_xlim(0.16, 0.285); b.set_ylim(len(rows) - 0.4, -0.9)
    b.grid(axis="x", color=GRID, lw=0.5)
    for yi, (u, v) in enumerate(zip(c21, c26)):
        b.plot([v, u], [yi, yi], color=AXIS, lw=1.6, solid_capstyle="round", zorder=2)
    b.scatter(c21, y, s=40, color=CAT[1], zorder=4, edgecolor="white", linewidth=0.8)
    b.scatter(c26, y, s=16, color=CAT[0], zorder=5, edgecolor="white", linewidth=0.6)
    i = [k for k, s in enumerate(rows) if s["L2"] == "Hetzner CX23 (shared)"][0]
    b.text(c26[i] - 0.003, i, "%.3f" % c26[i], ha="right", va="center", fontsize=7, color=INK)
    b.text(c21[i] + 0.004, i, "%.3f" % c21[i], ha="left", va="center", fontsize=7, color=INK)
    b.text((c21[i] + c26[i]) / 2, i - 0.42, "−%.0f%%" % ((1 - c26[i] / c21[i]) * 100),
           ha="center", va="bottom", fontsize=7, color=INK2)
    b.text(c21[0] + 0.004, 0, "%.3f (all)" % c21[0], ha="left", va="center", fontsize=7, color=INK)
    b.set_yticks(y); b.set_yticklabels([lab[s["L2"]] for s in rows])
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    b.xaxis.set_major_locator(FixedLocator([0.17, 0.20, 0.23, 0.26]))
    b.set_xlabel("Optimal cost (USD per million requests)")
    b.set_title("b  Cost by external offer")
    hl = [Line2D([], [], marker="o", ls="", color=CAT[1], markersize=6, label="2021 rule"),
          Line2D([], [], marker="o", ls="", color=CAT[0], markersize=4,
                 label="2026 rule, EU destination listed")]
    b.legend(handles=hl, loc="lower left", bbox_to_anchor=(-0.02, -0.33), ncol=2,
             handletextpad=0.3, columnspacing=1.2, fontsize=7)
    save(fig, "v3_fig1_placement")


# =========================================================================
# Figure 2 - spectra and exact rank
# =========================================================================
def figure2():
    corp = [("ml-1m", "MovieLens-1M"), ("gowalla", "Gowalla"), ("yelp2018", "Yelp2018"),
            ("amazon-book", "Amazon-Book")]
    series = []
    for c, lab_ in corp:
        o = json.load(open(os.path.join(V2, "spectra_%s.json" % c)))
        p = o["by_round"][str(o["rounds"])]
        series.append((lab_, np.array(p["sv"]), o["n_item"], p["zeta_all"]))
    old = json.load(open(os.path.join(ROOT, "real", "natijalar", "spektr_toliq.json")))["oxirgi_raund"]
    series.append(("Video Games (v1)", np.array(old["sv_barchasi"]), 2048, old["power_law_zeta"]))
    cols = CAT[:5]

    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 70 * MM),
                               gridspec_kw={"wspace": 0.95, "width_ratios": [1.0, 0.9]})
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.80)
    k = np.arange(1, 65)
    ref = k ** (-1.0)
    a.plot(k, ref, color=MUTED, lw=0.9, ls=(0, (1, 1.6)), zorder=1)
    ends = []
    for (lab_, s, d1, z), c in zip(series, cols):
        ls = (0, (4, 2)) if "v1" in lab_ else "-"
        a.plot(k, s / s[0], color=c, lw=1.4, ls=ls, zorder=3)
        ends.append((np.log10(s[-1] / s[0]), lab_, z, c))
    ends.sort()
    pos = spread([e[0] for e in ends], 0.13)
    for (yv, lab_, z, c), yp in zip(ends, pos):
        a.text(70, 10 ** yp, "%s  (ζ = %s)" % (lab_.replace(" (v1)", " v1"), ("%.2f" % z).replace("-", "−")),
               color=INK, fontsize=6.8, va="center", ha="left")
        a.plot([64.5, 68.5], [10 ** yv, 10 ** yp], color=c, lw=0.7)
    a.text(40, 1 / 40 * 0.62, "power law, ζ = 1", color=INK2, fontsize=6.8, ha="center", va="top")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(1, 64); a.set_ylim(0.012, 1.15)
    a.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64])); a.xaxis.set_major_formatter(plain)
    a.yaxis.set_major_locator(FixedLocator([0.01, 0.03, 0.1, 0.3, 1])); a.yaxis.set_major_formatter(plain)
    a.xaxis.set_minor_locator(NullLocator()); a.yaxis.set_minor_locator(NullLocator())
    a.grid(True, which="major", color=GRID, lw=0.5)
    a.set_xlabel(r"Singular-value index  $j$"); a.set_ylabel(r"Normalized singular value  $s_j / s_1$")
    a.set_title("a  Spectrum of the clipped aggregate")
    a.set_clip_on(False)

    snr = np.geomspace(0.3, 30, 400)
    for (lab_, s, d1, z), c in zip(series, cols):
        sig = s[0] / (snr * np.sqrt(d1))
        rs = [dp.optimal_rank_threshold(s ** 2, d1, x) for x in sig]
        ls = (0, (4, 2)) if "v1" in lab_ else "-"
        b.step(snr, rs, where="post", color=c, lw=1.4, ls=ls)
    b.set_xscale("log"); b.set_yscale("log", base=2)
    b.set_xlim(0.3, 30); b.set_ylim(0.85, 80)
    b.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30])); b.xaxis.set_major_formatter(plain)
    b.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64])); b.yaxis.set_major_formatter(plain)
    b.xaxis.set_minor_locator(NullLocator()); b.yaxis.set_minor_locator(NullLocator())
    b.grid(True, which="major", color=GRID, lw=0.5)
    b.set_xlabel(r"Signal-to-noise ratio  $s_1 / (\sigma\, d_1^{1/2})$")
    b.set_ylabel(r"Optimal rank  $r^{*}$  (Proposition 1)")
    b.set_title("b  Exact optimal rank")
    b.text(0.33, 64 * 0.82, "rank ceiling = 64", fontsize=6.8, color=INK2, va="top")

    hl = [Line2D([], [], color=c, lw=1.6, ls=((0, (4, 2)) if "v1" in s_[0] else "-"), label=s_[0])
          for s_, c in zip(series, cols)]
    fig.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5,
               handlelength=2.4, columnspacing=1.4, fontsize=7.3)
    save(fig, "v3_fig2_spectra")


# =========================================================================
# Figure 3 - downstream
# =========================================================================
def figure3():
    D = json.load(open(os.path.join(V2, "downstream_ml-1m.json")))
    eps_list = sorted({g["eps_total"] for g in D["grid"]})
    ranks = sorted({g["rank"] for g in D["grid"]})
    n_seed = len(D["seeds"])
    fig, ax = plt.subplots(figsize=(W_FULL * 0.78, 78 * MM))
    fig.subplots_adjust(left=0.1, right=0.74, bottom=0.15, top=0.97)
    npv = np.mean([x["recall20"] for x in D["nonprivate"]["full_rank"]])
    pop = D["popularity"]["recall20"]
    ax.axhline(npv, color=INK, lw=0.9, ls=(0, (5, 2.5)), zorder=1)
    ax.axhline(pop, color=MUTED, lw=0.9, ls=(0, (1, 1.6)), zorder=1)
    x = np.arange(len(ranks))
    offs = np.linspace(-0.12, 0.12, len(eps_list))
    for e, c, dx in zip(eps_list, RAMP, offs):
        cells = [[g["recall20"] for g in D["grid"] if g["eps_total"] == e and g["rank"] == r] for r in ranks]
        mu = np.array([np.mean(v) for v in cells]); sd = np.array([np.std(v, ddof=1) for v in cells])
        xx = x + dx
        ax.errorbar(xx, mu, yerr=sd, fmt="none", ecolor=c, elinewidth=0.9, capsize=0, zorder=2)
        ax.plot(xx, mu, color=c, lw=1.6, zorder=3)
        ax.scatter(xx, mu, s=16, color=c, zorder=4, edgecolor="white", linewidth=0.6)
        i = int(np.argmax(mu))
        ax.scatter([xx[i]], [mu[i]], s=80, facecolor="none", edgecolor=c, linewidth=1.3, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(ranks)
    ax.set_xlim(-0.25, len(ranks) - 0.75)
    ax.set_ylim(0.0, 0.11)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.02, 0.04, 0.06, 0.08, 0.10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%.2f" % v))
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_xlabel(r"Transmitted rank  $r$")
    ax.set_ylabel("Recall@20 (MovieLens-1M)")
    # right-margin labels: references and the legend for the budgets
    tr = ax.get_yaxis_transform()
    ax.text(1.02, npv, "non-private,\nfull rank", transform=tr, fontsize=7, va="center",
            ha="left", color=INK, linespacing=1.1)
    ax.text(1.02, pop, "popularity", transform=tr, fontsize=7, va="center", ha="left", color=INK2)
    hl = [Line2D([], [], color=c, lw=1.6, marker="o", markersize=3.5, markeredgecolor="white",
                 markeredgewidth=0.5, label="%g" % e) for e, c in zip(eps_list, RAMP)][::-1]
    hl.append(Line2D([], [], ls="", marker="o", markersize=7, markerfacecolor="none",
                     markeredgecolor=INK2, markeredgewidth=1.1, label="best rank"))
    hl.append(Line2D([], [], color=INK2, ls="", marker="|", markersize=9, markeredgewidth=0.9, label="± 1 s.d."))
    ax.legend(handles=hl, loc="upper left", bbox_to_anchor=(1.02, 0.60),
              title=r"Total budget $\varepsilon_{\mathrm{tot}}$", title_fontsize=7.3,
              fontsize=7.3, handlelength=2.0, labelspacing=0.45, alignment="left")
    save(fig, "v3_fig3_downstream")


if __name__ == "__main__":
    figure1(); figure2(); figure3()
