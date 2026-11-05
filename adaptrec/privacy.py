# -*- coding: utf-8 -*-
"""A6 — maxfiylik byudjetini taqsimlash va past rangli siqish
(dissertatsiya, 3.4-bo'lim).

Uch o'lchovli bog'liqlik (2.4-bo'lim): maxfiylik byudjeti epsilon
kichraysa, optimal siqish darajasi ham KUCHAYADI:
    r* ~ epsilon^(2 / (zeta + 1))
ya'ni past rangli proyeksiya bir vaqtning o'zida kommunikatsiyani ham,
DP shovqinining ta'sirini ham kamaytiradi (CoLR yo'nalishining
nazariy umumlashmasi).
"""
import numpy as np


def optimal_rang(eps, r_max, zeta=1.0, c0=1.0):
    """Berilgan maxfiylik byudjetida optimal past rang darajasi."""
    r = c0 * float(eps) ** (2.0 / (float(zeta) + 1.0)) * r_max
    return int(np.clip(round(r), 1, r_max))


def byudjet_taqsimoti(eps_umumiy, T, siyosat="teng", tezlashish=2.0):
    """Raundlar bo'yicha epsilon taqsimoti.

    "teng"     — har raundga eps/T;
    "kamayuvchi" — dastlabki raundlarga ko'proq (model tez shakllanadi),
                 geometrik kamayish, yig'indisi eps_umumiy ga teng.
    """
    T = int(T)
    if siyosat == "teng":
        return np.full(T, eps_umumiy / T)
    q = float(tezlashish) ** (-1.0 / max(T - 1, 1))
    og = q ** np.arange(T)
    return eps_umumiy * og / og.sum()


def gauss_shovqin_sigma(eps, delta, sezgirlik=1.0):
    """Gauss mexanizmi uchun sigma (klassik baho)."""
    return sezgirlik * np.sqrt(2.0 * np.log(1.25 / delta)) / eps


def past_rang_proyeksiya(G, r, rng=None):
    """Gradiyent/yangilanish matritsasini r-rangli fazoga proyeksiyalash.
    Qaytaradi: (U_r, V_r) — uzatiladigan siqilgan ko'rinish."""
    U, s, Vt = np.linalg.svd(np.asarray(G, float), full_matrices=False)
    r = int(min(r, len(s)))
    return U[:, :r] * s[:r], Vt[:r]


def dp_past_rang_yangilanish(G, r, eps, delta, klip=1.0, rng=None):
    """Past rangli siqish + DP shovqin: A6 ning bitta raunddagi qadami."""
    rng = rng or np.random.default_rng()
    A, B = past_rang_proyeksiya(G, r)
    nrm = np.linalg.norm(A)
    if nrm > klip:
        A = A * (klip / nrm)
    sigma = gauss_shovqin_sigma(eps, delta, sezgirlik=klip)
    A_shovqinli = A + rng.normal(0.0, sigma, A.shape)
    return A_shovqinli, B
