# -*- coding: utf-8 -*-
"""Revised Section 5 checks.

1. Noise calibration: classical Gaussian bound (valid only for eps < 1),
   analytic Gaussian (Balle & Wang 2018) and zCDP composition over T rounds.
2. Sensitivity of the ORIGINAL mechanism (V_r taken from the SVD of the
   private aggregate) versus the revised one (V fixed before the round):
   on a real aggregate of clipped client updates we remove one client at a
   time and measure ||A(G) - A(G')||_F, which a Gaussian mechanism with
   sensitivity C requires to be <= C. The original mechanism also releases
   V_r itself without noise; we report how far V_r moves when one client is
   removed (principal-angle sine), i.e. what that unprotected release leaks.

Run from the package root:  python experiments/v2_mechanism.py
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from adaptrec import fedmf, dp

OUT = os.path.join(ROOT, "natijalar", "v2", "mechanism.json")
res = {}

# 1. calibration -------------------------------------------------------------
delta = 1e-5
rows = []
for eps in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 96):
    rows.append({"eps": eps,
                 "classical_sigma": dp.classical_gaussian_sigma(eps, delta),
                 "analytic_sigma": dp.analytic_gaussian_sigma(eps, delta),
                 "zcdp_sigma_T1": dp.sigma_for_total_budget(eps, delta, 1),
                 "zcdp_sigma_T50": dp.sigma_for_total_budget(eps, delta, 50),
                 "eps_total_if_per_round_T50_zcdp":
                     dp.eps_from_rho(50 * 1.0 / (2 * dp.analytic_gaussian_sigma(eps, delta) ** 2),
                                     delta)})
res["calibration"] = rows
for r in rows:
    print("eps=%-5g classical=%.3f analytic=%.3f zCDP(T=1)=%.3f zCDP(T=50 total)=%.3f  "
          "per-round eps over 50 rounds -> total %.0f" % (
              r["eps"], r["classical_sigma"], r["analytic_sigma"], r["zcdp_sigma_T1"],
              r["zcdp_sigma_T50"], r["eps_total_if_per_round_T50_zcdp"]))

# 2. sensitivity ---------------------------------------------------------------
CLIP, R_TRAIN, N_REMOVE = 2.0, 50, 300
data = fedmf.Data("ml-1m")
tr = fedmf.train(data, rounds=R_TRAIN, lr_item=50.0, clip=CLIP, seed=0)
P, Q = tr["P"], tr["Q"]
rng = np.random.default_rng(5)
users = np.where(data.deg > 0)[0]
Cm = fedmf.client_round(data, P.copy(), Q, users, 16, 0.0, 0.0, rng)
cn = np.sqrt(np.asarray(Cm.multiply(Cm).sum(1)).ravel())
nrm = cn * np.linalg.norm(P, axis=1)
sc = np.where(nrm > 0, np.minimum(1.0, CLIP / np.maximum(nrm, 1e-300)), 0.0)
W = sc[:, None] * P
G = np.asarray(Cm.T @ W)
_, s, vt = np.linalg.svd(G, full_matrices=False)
Vfix = dp.random_basis(G.shape[1], np.random.default_rng(123))
order = np.argsort(-nrm * sc)                  # clients with the largest clipped updates
cand = order[:N_REMOVE]
sens = {r: {"orig": [], "fixed": [], "sin_angle": []} for r in (1, 4, 16)}
for u in cand:
    Gu = np.outer(np.asarray(Cm[u].todense()).ravel(), W[u])
    G2 = G - Gu
    _, s2, vt2 = np.linalg.svd(G2, full_matrices=False)
    for r in sens:
        A1 = G @ vt[:r].T; A2 = G2 @ vt2[:r].T
        # align signs of singular vectors (sign is arbitrary in the SVD)
        sg = np.sign(np.sum(vt[:r] * vt2[:r], axis=1)); A2 = A2 * sg
        sens[r]["orig"].append(float(np.linalg.norm(A1 - A2)))
        sens[r]["fixed"].append(float(np.linalg.norm((G - G2) @ Vfix[:, :r])))
        M = vt[:r] @ vt2[:r].T
        cos_min = np.linalg.svd(M, compute_uv=False).min()
        sens[r]["sin_angle"].append(float(np.sqrt(max(0.0, 1 - cos_min ** 2))))
res["sensitivity"] = {"clip": CLIP, "clients_tested": int(N_REMOVE),
                      "top_singular_values": [float(x) for x in s[:6]]}
for r, v in sens.items():
    res["sensitivity"]["rank_%d" % r] = {
        "orig_max": max(v["orig"]), "orig_median": float(np.median(v["orig"])),
        "orig_frac_above_C": float(np.mean(np.array(v["orig"]) > CLIP)),
        "fixed_max": max(v["fixed"]),
        "max_sin_angle_Vr": max(v["sin_angle"])}
    print("rank %2d: original max ||dA||=%.3f (%.0f%% of removals exceed C=%.1f); "
          "fixed-basis max=%.3f; max sin-angle of V_r=%.4f" % (
              r, max(v["orig"]), 100 * np.mean(np.array(v["orig"]) > CLIP), CLIP,
              max(v["fixed"]), max(v["sin_angle"])))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT)
