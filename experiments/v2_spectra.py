# -*- coding: utf-8 -*-
"""Revised Section 6.2: spectra of federated update matrices across corpora
and rounds, with client-bootstrap confidence intervals.

For each corpus a non-private federated BPR-MF run (per-client clipping at C)
is executed; at the recorded rounds a fresh round of client updates is drawn
from the current model and
  * the singular values of the clipped aggregate G (n_items x 64) are taken;
  * the power-law exponent zeta is fitted on indices 1..64 and 2..64;
  * the share of energy in the leading direction is recorded;
  * the exact optimal rank r*(sigma) = argmin_r L(r) (Proposition 1) is
    evaluated on a sigma grid, and its log-log slope against 1/sigma over the
    interior range 2 <= r* <= 32 is compared with 2/(zeta+1);
  * every quantity is recomputed on N_REP independent draws of the round
    (fresh positive/negative pairs for every client at the same model state),
    giving 95% percentile intervals for round-to-round variability.
    A multinomial client bootstrap is NOT used: duplicating clients inflates
    the random-matrix bulk of the spectrum relative to its spikes and biases
    zeta downward (the point estimate fell outside its own bootstrap interval
    in a first run).

Run from the package root:  python experiments/v2_spectra.py [corpus ...]
"""
import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from adaptrec import fedmf, dp

CLIP = float(os.environ.get("CLIP", "2.0"))
SEED = int(os.environ.get("SEED", "0"))
TAG = os.environ.get("TAG", "")
LR_ITEM, D = 50.0, 64
N_REP = int(os.environ.get("N_REP", "50"))
PLAN = {"ml-1m": 200, "gowalla": 100, "yelp2018": 100, "amazon-book": 100}
SIGMA_GRID = np.geomspace(1e-3, 1e3, 241)


def fit_zeta(s, lo):
    k = np.arange(1, len(s) + 1)[lo:]
    y, x = np.log(s[lo:]), np.log(k)
    q, c = np.polyfit(x, y, 1)
    r2 = 1 - (y - (q * x + c)).var() / y.var()
    q2, c2 = np.polyfit(k, y, 1)
    r2e = 1 - (y - (q2 * k + c2)).var() / y.var()
    return -2 * q - 1, r2, r2e


def rank_curve(s, d1):
    e = s ** 2
    return np.array([dp.optimal_rank_threshold(e, d1, sg) for sg in SIGMA_GRID])


def slope_interior(rs):
    m = (rs >= 2) & (rs <= 32)
    if m.sum() < 5:
        return float("nan")
    return float(np.polyfit(np.log(1 / SIGMA_GRID[m]), np.log(rs[m]), 1)[0])


def summarise(s, d1):
    z_all, r2_all, r2e_all = fit_zeta(s, 0)
    z_t, r2_t, r2e_t = fit_zeta(s, 1)
    rs = rank_curve(s, d1)
    return {"zeta_all": z_all, "r2_powerlaw_all": r2_all, "r2_exp_all": r2e_all,
            "zeta_tail": z_t, "r2_powerlaw_tail": r2_t, "r2_exp_tail": r2e_t,
            "top_energy_share": float(s[0] ** 2 / (s ** 2).sum()),
            "slope_measured": slope_interior(rs),
            "slope_pred_all": 2 / (z_all + 1), "slope_pred_tail": 2 / (z_t + 1)}


def one_draw(data, P, Q, users, rng):
    Cm = fedmf.client_round(data, P.copy(), Q, users, 16, 0.0, 0.0, rng)
    G, frac, _ = fedmf.aggregate(Cm, P, CLIP)
    return np.linalg.svd(G, compute_uv=False), G.shape[0], frac


def analyse(data, P, Q, rng):
    users = np.where(data.deg > 0)[0]
    s, d1, frac = one_draw(data, P, Q, users, rng)
    point = summarise(s, d1)
    point["sv"] = [float(x) for x in s]
    point["clipped_frac"] = frac
    reps = [summarise(one_draw(data, P, Q, users, rng)[0], d1) for _ in range(N_REP)]
    ci = {}
    for k in reps[0]:
        v = np.array([x[k] for x in reps], float)
        v = v[np.isfinite(v)]
        ci[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    point["ci95"] = ci
    return point


def run(corpus):
    data = fedmf.Data(corpus)
    R = PLAN[corpus]
    rec_at = sorted({1, 10, R // 4, R // 2, R})
    rng = np.random.default_rng([11, len(corpus)] + ([SEED] if SEED else []))
    out = {"corpus": corpus, "n_user": data.n_user, "n_item": data.n_item,
           "n_train": int(data.n_train), "rounds": R, "clip": CLIP, "seed": SEED,
           "n_rep": N_REP, "rounds_analysed": rec_at, "by_round": {}}
    t0 = time.time()

    def cb(t, P, Q, G, Gh):
        if t + 1 in rec_at:
            out["by_round"][str(t + 1)] = analyse(data, P, Q, rng)
            p = out["by_round"][str(t + 1)]
            print("%s round %d: zeta_all=%.3f zeta_tail=%.3f top=%.2f slope=%.2f "
                  "(pred %.2f / %.2f)  %.0fs" % (corpus, t + 1, p["zeta_all"], p["zeta_tail"],
                  p["top_energy_share"], p["slope_measured"], p["slope_pred_all"],
                  p["slope_pred_tail"], time.time() - t0), flush=True)

    r = fedmf.train(data, rounds=R, lr_item=LR_ITEM, clip=CLIP, seed=SEED, callback=cb)
    out["final_recall20"], out["final_ndcg20"] = r["recall20"], r["ndcg20"]
    out["popularity"] = dict(zip(("recall20", "ndcg20"), fedmf.popularity(data)))
    path = os.path.join(ROOT, "natijalar", "v2", "spectra_%s%s.json" % (corpus, TAG))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print("saved", path, flush=True)


if __name__ == "__main__":
    for c in (sys.argv[1:] or list(PLAN)):
        run(c)
