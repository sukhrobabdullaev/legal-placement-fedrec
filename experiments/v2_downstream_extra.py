# -*- coding: utf-8 -*-
"""Adds seeds 3 and 4 to every cell of v2_downstream_dp.py (same settings) and
writes the merged grid back, so every (eps_total, rank) cell has five seeds.
Run from the package root after v2_downstream_dp.py."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from adaptrec import fedmf, dp
P = os.path.join(ROOT, "natijalar", "v2", "downstream_ml-1m.json")
D = json.load(open(P))
data = fedmf.Data(D["corpus"])
NEW = [3, 4]
for seed in NEW:
    if seed in D["seeds"]:
        continue
    r = fedmf.train(data, rounds=D["rounds"], lr_item=D["lr_item"], clip=D["clip"], seed=seed)
    D["nonprivate"]["full_rank"].append({"seed": seed, "recall20": r["recall20"], "ndcg20": r["ndcg20"]})
    for eps in sorted({g["eps_total"] for g in D["grid"]}):
        sigma = dp.sigma_for_total_budget(eps, D["delta"], D["rounds"], D["clip"])
        for rank in sorted({g["rank"] for g in D["grid"]}):
            r = fedmf.train(data, rounds=D["rounds"], lr_item=D["lr_item"], clip=D["clip"],
                            rank=rank, sigma=sigma, seed=seed)
            D["grid"].append({"eps_total": eps, "sigma": sigma, "rank": rank, "seed": seed,
                              "recall20": r["recall20"], "ndcg20": r["ndcg20"]})
        print("seed", seed, "eps", eps, "done", flush=True)
    D["seeds"].append(seed)
    json.dump(D, open(P, "w"), indent=1)
print("seeds now", D["seeds"])
