# -*- coding: utf-8 -*-
"""Revised Section 4.4, last paragraph: choosing the rank every round by
applying Proposition 1 to energies estimated from the previous public release
(rank="auto" in adaptrec.fedmf.train), compared with the best fixed rank of
v2_downstream_dp.py. Same corpus, rounds, clipping, budgets and seeds.

Run from the package root after v2_downstream_dp.py."""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from adaptrec import fedmf, dp

D = json.load(open(os.path.join(ROOT, "natijalar", "v2", "downstream_ml-1m.json")))
data = fedmf.Data("ml-1m")
out = {"rule": "per-round Proposition 1 on energies estimated from the previous release",
       "rows": []}
for eps in sorted({g["eps_total"] for g in D["grid"]}):
    sigma = dp.sigma_for_total_budget(eps, D["delta"], D["rounds"], D["clip"])
    rec, ranks = [], []
    for seed in D["seeds"]:
        r = fedmf.train(data, rounds=D["rounds"], lr_item=D["lr_item"], clip=D["clip"],
                        rank="auto", sigma=sigma, seed=seed)
        rec.append(r["recall20"]); ranks += r["ranks"]
    best = max({g["rank"] for g in D["grid"]},
               key=lambda q: np.mean([g["recall20"] for g in D["grid"]
                                      if g["eps_total"] == eps and g["rank"] == q]))
    bm = float(np.mean([g["recall20"] for g in D["grid"] if g["eps_total"] == eps and g["rank"] == best]))
    out["rows"].append({"eps_total": eps, "auto_recall20": float(np.mean(rec)),
                        "auto_rank_min": int(min(ranks)), "auto_rank_max": int(max(ranks)),
                        "auto_rank_mean": float(np.mean(ranks)),
                        "best_fixed_rank": int(best), "best_fixed_recall20": bm})
    print(out["rows"][-1], flush=True)
json.dump(out, open(os.path.join(ROOT, "natijalar", "v2", "auto_rank.json"), "w"), indent=1)
