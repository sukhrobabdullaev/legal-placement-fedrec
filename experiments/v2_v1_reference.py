# -*- coding: utf-8 -*-
"""Quantities about the first version (v1) that the revised article cites,
recomputed from the v1 outputs stored in real/natijalar/ and stored here so
that no number in the article lives only in the manuscript build script.

  * crossover eps above which the classical Gaussian bound gives LESS noise
    than the analytic calibration (i.e. under-noises), delta = 1e-5;
  * composition over a hypothetical 50 rounds of the v1 per-round budgets
    (4 and 96) under zCDP, with classical and with analytic per-round sigma;
  * the v1 Video Games spectrum: zeta with and without s_1, energy share of
    s_1, and the slope of the v1 rank sweep with its t-interval (8 points).
Run from the package root."""
import json, os, sys
import numpy as np
from scipy import stats
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from adaptrec import dp

delta = 1e-5
f = lambda e: dp.classical_gaussian_sigma(e, delta) - dp.analytic_gaussian_sigma(e, delta)
cross = brentq(f, 2.0, 32.0)
comp = {}
for e in (4.0, 96.0):
    for name, sig in (("classical", dp.classical_gaussian_sigma(e, delta)),
                      ("analytic", dp.analytic_gaussian_sigma(e, delta))):
        comp["%s_per_round_%g" % (name, e)] = dp.eps_from_rho(50 * 1.0 / (2 * sig ** 2), delta)

sp = json.load(open(os.path.join(ROOT, "real", "natijalar", "spektr_toliq.json")))["oxirgi_raund"]
s = np.array(sp["sv_barchasi"]); k = np.arange(1, len(s) + 1)
q_all = np.polyfit(np.log(k), np.log(s), 1)[0]
q_tail = np.polyfit(np.log(k[1:]), np.log(s[1:]), 1)[0]
z = json.load(open(os.path.join(ROOT, "real", "natijalar", "zeta_rekonstruksiya_zich.json")))
eps = np.array(z["eps_grid"]); rs = np.array(z["r_yulduz"], float)
m = (rs > 1) & (rs < 64)
res = stats.linregress(np.log(eps[m]), np.log(rs[m]))
t = stats.t.ppf(0.975, m.sum() - 2)
out = {"delta": delta, "classical_underestimates_above_eps": cross, "composition_50_rounds": comp,
       "video_games_v1": {"zeta_all": float(-2 * q_all - 1), "zeta_tail": float(-2 * q_tail - 1),
                          "top_energy_share": float(s[0] ** 2 / (s ** 2).sum()),
                          "power_law_R2": sp["power_law_R2"], "exp_R2": sp["eksponensial_R2"],
                          "slope": float(res.slope), "slope_t95": [float(res.slope - t * res.stderr),
                                                                   float(res.slope + t * res.stderr)],
                          "n_points": int(m.sum()), "pred_slope": float(2 / (-2 * q_all - 1 + 1))}}
json.dump(out, open(os.path.join(ROOT, "natijalar", "v2", "v1_reference.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
