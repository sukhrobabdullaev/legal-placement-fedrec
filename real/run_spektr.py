# -*- coding: utf-8 -*-
"""Rang qonunini REAL update matritsasida rekonstruksiya mezoni bo'yicha
tekshirish, plus spektr shaklini xarakterlash.

Downstream NDCG varianti shovqin bilan to'lgan edi, chunki MF asosi zaif.
Rekonstruksiya mezoni bu chalkashlikni olib tashlaydi va sintetik
tajriba bilan AYNAN bir xil mezon bo'ladi, ya'ni to'g'ridan-to'g'ri
solishtiriladi.
"""
import sys, os, json, time, collections
import numpy as np
import os as _os, sys as _sys
# 1-maqola paketidagi real/ katalogi (Oqim, komponentlar) shu yerdan olinadi.
_M1 = _os.environ.get("MAQOLA1_REAL") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                  "..", "..", "Maqola1_kodlari", "real"))
if not _os.path.isdir(_os.path.join(_M1, "adaptreal")):
    raise SystemExit("1-maqola real/ katalogi topilmadi: " + _M1 +
                     "  (MAQOLA1_REAL muhit o'zgaruvchisi bilan ko'rsating)")
_sys.path.insert(0, _M1)
from adaptreal.data import Oqim

D, EPOCH, SILO, RAUND = 64, 40, 2000, 6
N_ITEM_UPD = 2048
DELTA, CLIP = 1e-5, 1.0
N_REP = 60
rng = np.random.default_rng(7)

s = Oqim(); n_u, n_i, N = s.n_user, s.n_item, len(s)
kesim = int(N * 0.9); tr_u, tr_i = s.u[:kesim], s.i[:kesim]
P = (rng.standard_normal((n_u, D)) * 0.05).astype(np.float32)
Q = (rng.standard_normal((n_i, D)) * 0.05).astype(np.float32)
lr, reg, B = 0.08, 0.005, 8192
for ep in range(EPOCH):
    perm = rng.permutation(kesim)
    for b0 in range(0, kesim, B):
        idx = perm[b0:b0+B]; u, i = tr_u[idx], tr_i[idx]
        j = rng.integers(0, n_i, len(idx)).astype(np.int32)
        pu, qi, qj = P[u], Q[i], Q[j]
        x = np.einsum("bd,bd->b", pu, qi - qj)
        g = (1.0 / (1.0 + np.exp(x)))[:, None].astype(np.float32)
        np.add.at(P, u, lr * (g * (qi - qj) - reg * pu))
        np.add.at(Q, i, lr * (g * pu - reg * qi))
        np.add.at(Q, j, lr * (-g * pu - reg * qj))
print("MF o'qitildi", flush=True)

faol = np.argsort(-np.bincount(tr_i, minlength=n_i))[:N_ITEM_UPD]
pos = -np.ones(n_i, np.int64); pos[faol] = np.arange(len(faol))
silo_of = rng.integers(0, SILO, n_u)
bloklar = np.array_split(np.arange(kesim), RAUND)
Gs = []
for r_ in range(RAUND):
    ev = bloklar[r_]; G = np.zeros((len(faol), D), np.float32)
    uu, ii = tr_u[ev], tr_i[ev]; m0 = pos[ii] >= 0; uu, ii = uu[m0], ii[m0]
    sid = silo_of[uu]; o = np.argsort(sid, kind="stable")
    uu, ii, sid = uu[o], ii[o], sid[o]
    ch = np.searchsorted(sid, np.arange(SILO + 1))
    for k in range(SILO):
        a, b = ch[k], ch[k+1]
        if a == b: continue
        u2, i2 = uu[a:b], ii[a:b]
        j2 = faol[rng.integers(0, len(faol), len(u2))]
        pu, qi, qj = P[u2], Q[i2], Q[j2]
        x = np.einsum("bd,bd->b", pu, qi - qj)
        g = (1.0/(1.0+np.exp(x)))[:, None].astype(np.float32)
        Gl = np.zeros((len(faol), D), np.float32)
        np.add.at(Gl, pos[i2], g*pu); np.add.at(Gl, pos[j2], -g*pu)
        nrm = np.linalg.norm(Gl)
        if nrm > CLIP: Gl *= CLIP/nrm
        G += Gl
    Gs.append(G)

spektr = {}
for nom, G in [("oxirgi_raund", Gs[-1])]:


    sv = np.linalg.svd(G, compute_uv=False); sv = sv[sv > 0]
    k = np.arange(1, len(sv)+1)
    y, x = np.log(sv), np.log(k)
    q1, c1 = np.polyfit(x, y, 1); r2_pl = 1 - (y-(q1*x+c1)).var()/y.var()
    q2, c2 = np.polyfit(k, y, 1); r2_exp = 1 - (y-(q2*k+c2)).var()/y.var()
    spektr[nom] = {"n_sv": int(len(sv)),
                   "power_law_zeta": round(float(-2*q1-1), 4),
                   "power_law_R2": round(float(r2_pl), 4),
                   "eksponensial_R2": round(float(r2_exp), 4),
                   "power_law_c": float(c1), "power_law_qiyalik": float(q1),
                   "exp_c": float(c2), "exp_qiyalik": float(q2),
                   "sv_barchasi": [round(float(v), 6) for v in sv],
                   "sv_top5": [round(float(v), 4) for v in sv[:5]],
                   "sv_nisbat_s1_s64": round(float(sv[0]/sv[-1]), 2)}
    print(nom, spektr[nom]["power_law_zeta"], spektr[nom]["power_law_R2"],
          spektr[nom]["eksponensial_R2"], flush=True)

json.dump(spektr, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
          "natijalar", "spektr_toliq.json"), "w"), indent=1)
print("spektr_toliq.json saqlandi")
