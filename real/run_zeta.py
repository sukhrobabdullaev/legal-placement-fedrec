# -*- coding: utf-8 -*-
"""Real federated update matritsalaridan zeta ni O'LCHASH (2-maqola).

Maqolada zeta FARAZ qilingan (0,5 / 1 / 2). Bu skript uni real
ma'lumotdan o'lchaydi va rang qonunini downstream tavsiya sifati
bo'yicha tekshiradi.

Uchta nozik joy (birinchi urinishda uchalasi ham xato edi):

  * MF yetarlicha o'qitilishi kerak. Har bir foydalanuvchida o'rtacha
    8,7 ta hodisa bor, shuning uchun bir necha epoch yetmaydi.
  * Shovqin (n, r) FAKTORIGA qo'shiladi, to'liq qayta tiklangan
    matritsaga emas. Aks holda shovqin rangga bog'liq bo'lmay qoladi
    va Teorema 2 ning butun mexanizmi yo'qoladi.
  * Silo soni real bo'lishi kerak. DP sezuvchanligi bitta siloning
    hissasi bilan belgilanadi, signal esa hammasining yig'indisi.
    Kam silo bilan shovqin signalni butunlay bosadi.
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

D = int(os.environ.get("D", "64"))
EPOCH = int(os.environ.get("EPOCH", "40"))
SILO = int(os.environ.get("SILO", "2000"))
RAUND = int(os.environ.get("RAUND", "6"))
N_ITEM_UPD = int(os.environ.get("N_ITEM_UPD", "2048"))
N_EVAL = int(os.environ.get("N_EVAL", "3000"))
DELTA, CLIP = 1e-5, 1.0
rng = np.random.default_rng(7)

s = Oqim()
n_u, n_i, N = s.n_user, s.n_item, len(s)
kesim = int(N * 0.9)
print("oqim: %d hodisa, %d user, %d item" % (N, n_u, n_i), flush=True)

P = (rng.standard_normal((n_u, D)) * 0.05).astype(np.float32)
Q = (rng.standard_normal((n_i, D)) * 0.05).astype(np.float32)
tr_u, tr_i = s.u[:kesim], s.i[:kesim]
lr, reg, B = 0.08, 0.005, 8192
T0 = time.time()
for ep in range(EPOCH):
    perm = rng.permutation(kesim)
    for b0 in range(0, kesim, B):
        idx = perm[b0:b0+B]
        u, i = tr_u[idx], tr_i[idx]
        j = rng.integers(0, n_i, len(idx)).astype(np.int32)
        pu, qi, qj = P[u], Q[i], Q[j]
        x = np.einsum("bd,bd->b", pu, qi - qj)
        g = (1.0 / (1.0 + np.exp(x)))[:, None].astype(np.float32)
        np.add.at(P, u, lr * (g * (qi - qj) - reg * pu))
        np.add.at(Q, i, lr * (g * pu - reg * qi))
        np.add.at(Q, j, lr * (-g * pu - reg * qj))
    if (ep + 1) % 10 == 0: print("  epoch %d, %.0fs" % (ep + 1, time.time()-T0), flush=True)

ev_u, ev_i = s.u[kesim:], s.i[kesim:]
sel = rng.choice(len(ev_u), size=min(N_EVAL, len(ev_u)), replace=False)
EU, EI = ev_u[sel], ev_i[sel]
korilgan = collections.defaultdict(set)
for a, b in zip(tr_u, tr_i): korilgan[int(a)].add(int(b))

def ndcg(Qm, blok=1024):
    tot = 0.0
    for b0 in range(0, len(EU), blok):
        uu, ii = EU[b0:b0+blok], EI[b0:b0+blok]
        S = P[uu] @ Qm.T
        for r_, a in enumerate(uu):
            k = korilgan.get(int(a))
            if k: S[r_, list(k)] = -1e30
        tv = S[np.arange(len(ii)), ii]
        rank = 1 + (S > tv[:, None]).sum(1)
        tot += np.where(rank <= 10, 1.0 / np.log2(rank + 1.0), 0.0).sum()
    return float(tot / len(EU))

baza = ndcg(Q)
print("asosiy NDCG@10 = %.4f  (%.0fs)" % (baza, time.time()-T0), flush=True)

# ---- federated raundlar -----------------------------------------------
# Faol itemlar: real raundda faqat tegilgan itemlar yangilanadi
faol = np.argsort(-np.bincount(tr_i, minlength=n_i))[:N_ITEM_UPD]
pos = -np.ones(n_i, np.int64); pos[faol] = np.arange(len(faol))
silo_of = rng.integers(0, SILO, n_u)
bloklar = np.array_split(np.arange(kesim), RAUND)
Gs = []
for r_ in range(RAUND):
    ev = bloklar[r_]
    G = np.zeros((len(faol), D), np.float32)
    uu, ii = tr_u[ev], tr_i[ev]
    m0 = pos[ii] >= 0
    uu, ii = uu[m0], ii[m0]
    sid = silo_of[uu]
    tartib = np.argsort(sid, kind="stable")
    uu, ii, sid = uu[tartib], ii[tartib], sid[tartib]
    chegara = np.searchsorted(sid, np.arange(SILO + 1))
    for k in range(SILO):
        a, b = chegara[k], chegara[k+1]
        if a == b: continue
        u2, i2 = uu[a:b], ii[a:b]
        j2 = faol[rng.integers(0, len(faol), len(u2))]
        pu, qi, qj = P[u2], Q[i2], Q[j2]
        x = np.einsum("bd,bd->b", pu, qi - qj)
        g = (1.0 / (1.0 + np.exp(x)))[:, None].astype(np.float32)
        Gl = np.zeros((len(faol), D), np.float32)
        np.add.at(Gl, pos[i2], g * pu); np.add.at(Gl, pos[j2], -g * pu)
        nrm = np.linalg.norm(Gl)
        if nrm > CLIP: Gl *= CLIP / nrm     # silo darajasida kesish = DP sezuvchanligi
        G += Gl
    Gs.append(G)
    print("  raund %d: ||G|| = %.1f" % (r_ + 1, np.linalg.norm(G)), flush=True)

def zeta_ol(G):
    sv = np.linalg.svd(G, compute_uv=False); sv = sv[sv > 0]
    k = np.arange(1, len(sv) + 1)
    y, x = np.log(sv), np.log(k)
    qiya, kes = np.polyfit(x, y, 1)
    r2 = 1.0 - (y - (qiya*x + kes)).var() / y.var()
    return -2.0 * qiya - 1.0, float(r2)

zl = [zeta_ol(G) for G in Gs]
Z, Z_SD, R2 = float(np.mean([z for z,_ in zl])), float(np.std([z for z,_ in zl])), float(np.mean([r for _,r in zl]))
print("\nO'LCHANGAN zeta = %.3f +- %.3f  (fit R2 = %.4f)" % (Z, Z_SD, R2), flush=True)

# ---- (rang, eps) sweep, downstream NDCG -------------------------------
G = Gs[-1]
U, sv, Vt = np.linalg.svd(G, full_matrices=False)
n_row = G.shape[0]
# qo'llash qadami: yangilanish Q ning kichik, lekin sezilarli buzilishi bo'lsin
ALFA = float(os.environ.get("ALFA", "0")) or (0.15 * np.linalg.norm(Q) / max(np.linalg.norm(G), 1e-9))
print("qo'llash koeffitsiyenti alfa = %.4f" % ALFA, flush=True)
RANKS = list(range(1, D + 1))
EPS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
natija = {}
for eps in EPS:
    sig = CLIP * np.sqrt(2 * np.log(1.25 / DELTA)) / eps
    q = []
    for r_ in RANKS:
        A = U[:, :r_] * sv[:r_]                       # (n_row, r) uzatiladigan faktor
        Bm = Vt[:r_]
        An = A + rng.normal(0, sig, A.shape).astype(np.float32)   # shovqin FAKTORGA
        Q2 = Q.copy(); Q2[faol] += ALFA * (An @ Bm)
        q.append(ndcg(Q2))
    q = np.asarray(q); best = int(RANKS[int(np.argmax(q))])
    natija[str(eps)] = {"r_empirik": best, "NDCG_max": round(float(q.max()), 4),
                        "NDCG_r1": round(float(q[0]), 4), "NDCG_rD": round(float(q[-1]), 4)}
    print("  eps=%-5g r* = %-3d  NDCG %.4f  (r=1: %.4f, r=%d: %.4f)"
          % (eps, best, q.max(), q[0], D, q[-1]), flush=True)

rs = np.array([natija[str(e)]["r_empirik"] for e in EPS], float)
ich = (rs > 1) & (rs < D)
qiya = float(np.polyfit(np.log(np.array(EPS)[ich]), np.log(rs[ich]), 1)[0]) if ich.sum() > 2 else float("nan")
kutilgan = 2.0 / (Z + 1.0)
out = {"d": D, "epoch": EPOCH, "silo": SILO, "raund": RAUND, "n_item_upd": int(len(faol)),
       "n_eval": len(EU), "asosiy_NDCG": round(baza, 4), "alfa": round(float(ALFA), 5),
       "olchangan_zeta": round(Z, 4), "zeta_sd": round(Z_SD, 4), "zeta_fit_R2": round(R2, 4),
       "kutilgan_korsatkich": round(kutilgan, 4),
       "olchangan_qiyalik": round(qiya, 4) if qiya == qiya else None,
       "ichki_nuqta": int(ich.sum()), "eps_sweep": natija}
os.makedirs("natijalar", exist_ok=True)
json.dump(out, open("natijalar/zeta.json", "w"), ensure_ascii=False, indent=1)
print("\nkutilgan ko'rsatkich 2/(zeta+1) = %.3f, o'lchangan qiyalik = %s"
      % (kutilgan, out["olchangan_qiyalik"]), flush=True)
