# -*- coding: utf-8 -*-
"""2-maqola, 6.2-bo'lim: maxfiylik-siqish-foydalilik rang qonuni (Figure 2).

Nazariya (Teorema 2). Rank-r uzatish va kirish boshiga Gauss shovqini ostida
bir raundning kutilayotgan foydalilik yo'qotishi ikkiga ajraladi:

    E|| (A_r + N) B_r - G ||_F^2  =  sum_{i>r} s_i^2   +   d1 * r * sigma^2
                                      \_ kesish _/        \_ shovqin _/

Darajali spektr s_i = i^{-(zeta+1)/2} uchun kesish hadi ~ r^{-zeta}/zeta,
shovqin hadi esa r bo'yicha CHIZIQLI. Hosilani nolga tenglashtirsak:

    r* = (d1 * sigma^2)^{-1/(zeta+1)}   va   sigma ~ 1/eps
    =>  r* ~ eps^{2/(zeta+1)}                                    (5-formula)

TEKSHIRUV USULI. Qonunning mazmuni r* ning eps bo'yicha o'sish TARTIBIDA,
uchta alohida nuqtaning qiymatida emas. Shuning uchun bu skript:

  1. Har bir (rang, eps) nuqtasini mustaqil, takrorlanadigan RNG oqimi bilan
     baholaydi, ya'ni natija chaqirilish tartibiga bog'liq emas.
  2. eps ni 12 nuqtali logarifmik to'rda supurib, har birida empirik r* ni
     topadi, so'ng log-log qiyalikni baholab, uni nazariy 2/(zeta+1) bilan
     solishtiradi. Bu uchta zeta uchun takrorlanadi.
  3. Qiyalikni faqat CHEGARAVIY BO'LMAGAN nuqtalar bo'yicha baholaydi
     (1 < r* < d2). r* = 1 polida yoki r* = d2 shiftida turgan nuqtalar
     kesilgan bo'ladi va qiyalikni sun'iy ravishda pasaytiradi.

Ishga tushirish (paket ildizidan):  python experiments/rank_law.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C1, C2, C3 = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D9D9D9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 7
D1, D2 = 128, 64          # yangilanish matritsasi o'lchami
DELTA = 1e-5              # Gauss mexanizmining delta parametri
CLIP = 0.0065             # kesish normasi konstantasi
N_SWEEP = 150             # supurish uchun takrorlar soni
N_DETAIL = 400            # batafsil egri chiziqlar uchun takrorlar soni
RANKS = np.arange(1, D2 + 1)
ZETAS = (0.5, 1.0, 2.0)
EPS_SWEEP = np.geomspace(0.25, 64.0, 12)
EPS_DETAIL = [(1.0, C1, "-"), (4.0, C2, "--"), (16.0, C3, "-.")]


def spektr(zeta):
    """Darajali spektrli, deterministik yangilanish matritsasi."""
    g = np.random.default_rng([SEED, 1000 + int(zeta * 100)])
    Uq, _ = np.linalg.qr(g.normal(size=(D1, D2)))
    Vq, _ = np.linalg.qr(g.normal(size=(D2, D2)))
    sv = np.arange(1, D2 + 1) ** (-(zeta + 1) / 2.0)
    G = (Uq[:, :D2] * sv) @ Vq.T
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    return G, U, s, Vt


def sigma_of(eps):
    """Gauss mexanizmining kirish boshiga standart og'ishi."""
    return CLIP * np.sqrt(2 * np.log(1.25 / DELTA)) / eps


def xato(G, U, s, Vt, rank, eps, n_rep):
    """(o'rtacha nisbiy xato, standart xato).

    Urug' (rank, eps) dan hosil qilinadi: nuqta qiymati chaqirilish
    tartibidan mustaqil va bit darajasida takrorlanadi.
    """
    g = np.random.default_rng([SEED, int(rank), int(round(eps * 1e6))])
    A = U[:, :rank] * s[:rank]
    B = Vt[:rank]
    N = g.normal(0.0, sigma_of(eps), size=(n_rep, D1, rank))
    R = (A[None] + N) @ B - G[None]
    e = np.linalg.norm(R, axis=(1, 2)) / np.linalg.norm(G)
    return float(e.mean()), float(e.std(ddof=1) / np.sqrt(n_rep))


def kutilgan_egri(s, eps):
    """Yopiq shakldagi kutilma (nazariy egri chiziq, Monte-Karlosiz)."""
    nrm = np.linalg.norm(s)
    return np.array([np.sqrt(float((s[r:] ** 2).sum())
                             + D1 * r * sigma_of(eps) ** 2) / nrm for r in RANKS])


def egri(G, U, s, Vt, eps, n_rep):
    m = np.array([xato(G, U, s, Vt, int(r), eps, n_rep) for r in RANKS])
    mean, se = m[:, 0], m[:, 1]
    i = int(np.argmin(mean))
    tekis = RANKS[mean <= mean[i] + se[i]]      # minimumdan 1 SE ichidagi soha
    return mean, se, int(RANKS[i]), [int(x) for x in tekis]


# =========================================================================
# 1. Qonun ko'rsatkichi: r*(eps) log-log qiyaligi uch zeta uchun
# =========================================================================
qonun, sweep_xom = {}, {}
for zeta in ZETAS:
    G, U, s, Vt = spektr(zeta)
    rs = np.array([egri(G, U, s, Vt, float(e), N_SWEEP)[2] for e in EPS_SWEEP], float)
    ich = (rs > 1) & (rs < D2)                  # chegaraviy bo'lmagan nuqtalar
    if ich.sum() > 2:
        qiya, kes = np.polyfit(np.log(EPS_SWEEP[ich]), np.log(rs[ich]), 1)
        rezid = np.log(rs[ich]) - (qiya * np.log(EPS_SWEEP[ich]) + kes)
        r2 = 1.0 - rezid.var() / np.log(rs[ich]).var()
    else:
        qiya, kes, r2 = float("nan"), float("nan"), float("nan")
    qonun[str(zeta)] = {
        "kutilgan_korsatkich": round(2.0 / (zeta + 1.0), 4),
        "olchangan_qiyalik": round(float(qiya), 4),
        "fit_R2": round(float(r2), 4),
        "ichki_nuqtalar": int(ich.sum()),
        "r_yulduz": [int(x) for x in rs],
    }
    sweep_xom[str(zeta)] = (rs, ich, float(qiya), float(kes))

# =========================================================================
# 2. Figure 2
# =========================================================================
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.7))

# --- Chap panel: xato vs rang (zeta = 1), ishonch oralig'i bilan ----------
zeta = 1.0
G, U, s, Vt = spektr(zeta)
batafsil = {}
for eps, c, ls in EPS_DETAIL:
    mean, se, r_emp, tekis = egri(G, U, s, Vt, eps, N_DETAIL)
    teor = kutilgan_egri(s, eps)
    batafsil[str(eps)] = {"r_empirik": r_emp, "tekis_soha": tekis,
                          "r_nazariy": int(RANKS[int(np.argmin(teor))])}
    a.plot(RANKS, mean, color=c, ls=ls, lw=1.4, label=fr"$\varepsilon={eps:g}$")
    a.fill_between(RANKS, mean - 1.96 * se, mean + 1.96 * se, color=c, alpha=.18, lw=0)
    a.plot(RANKS, teor, color=c, lw=.7, alpha=.55)
    a.scatter([r_emp], [mean[r_emp - 1]], color=c, s=24, zorder=5)
a.set_xlabel(r"Transmitted rank $r$")
a.set_ylabel("Relative error")
a.set_xscale("log")
a.legend(frameon=False, fontsize=8)
a.set_title(r"(a) Trade-off at $\zeta=1$", fontsize=9, loc="left", pad=6)

# --- O'ng panel: o'lchangan r*(eps) va moslashtirilgan qiyalik ------------
for zeta, c, mk in zip(ZETAS, (C1, C2, C3), ("o", "s", "^")):
    rs, ich, qiya, kes = sweep_xom[str(zeta)]
    b.scatter(EPS_SWEEP[ich], rs[ich], color=c, s=20, marker=mk, zorder=5,
              label=fr"$\zeta={zeta:g}$: {qiya:.2f} (theory {2/(zeta+1):.2f})")
    b.scatter(EPS_SWEEP[~ich], rs[~ich], facecolors="none", edgecolors=c,
              s=20, marker=mk, alpha=.45, zorder=4)
    xx = np.geomspace(EPS_SWEEP[ich].min(), EPS_SWEEP[ich].max(), 40)
    b.plot(xx, np.exp(kes) * xx ** qiya, color=c, lw=1.2, alpha=.85)
b.axhline(D2, color=MUTED, lw=.7, ls=":")
b.set_xscale("log"); b.set_yscale("log")
b.set_xlabel(r"Privacy budget $\varepsilon$")
b.set_ylabel(r"Optimal rank $r^{*}$")
b.legend(frameon=False, fontsize=7.2, loc="lower right")
b.set_title(r"(b) Measured scaling of $r^{*}$", fontsize=9, loc="left", pad=6)

out = os.path.join(ROOT, "figures"); os.makedirs(out, exist_ok=True)
plt.savefig(os.path.join(out, "a2_fig2.png")); plt.close()

# =========================================================================
os.makedirs(os.path.join(ROOT, "natijalar"), exist_ok=True)
natija = {"geometriya": {"d1": D1, "d2": D2}, "seed": SEED,
          "n_rep_sweep": N_SWEEP, "n_rep_batafsil": N_DETAIL,
          "eps_sweep": [round(float(x), 4) for x in EPS_SWEEP],
          "qonun_korsatkichi": qonun,
          "zeta_1_batafsil": batafsil}
json.dump(natija, open(os.path.join(ROOT, "natijalar", "rank_law_results.json"), "w"),
          ensure_ascii=False, indent=1)
for z, v in qonun.items():
    print("zeta=%s  kutilgan=%.4f  olchangan=%.4f  R2=%.4f  (ichki nuqta: %d)"
          % (z, v["kutilgan_korsatkich"], v["olchangan_qiyalik"], v["fit_R2"],
             v["ichki_nuqtalar"]))
print("zeta=1 batafsil:", json.dumps(batafsil, ensure_ascii=False))
