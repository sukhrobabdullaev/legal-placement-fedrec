# -*- coding: utf-8 -*-
"""Revised Section 6.3: the rank trade-off measured by recommendation quality
under a correct, end-to-end private mechanism.

Federated BPR-MF on MovieLens-1M (each of the 6,040 users is a client),
per-client clipping at C, private low-rank release with a basis taken from
the previous public release (adaptrec.dp), zCDP accounting over all rounds.
For every total budget eps_total and transmitted rank r we train from scratch
and report Recall@20 / NDCG@20 on the held-out (leave-one-out) test item.

Run from the package root:  python experiments/v2_downstream_dp.py
"""
import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from adaptrec import fedmf, dp

CORPUS = os.environ.get("CORPUS", "ml-1m")
ROUNDS, CLIP, DELTA, LR_ITEM = 50, 2.0, 1e-5, 50.0
EPS = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
RANKS = [1, 2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2]
OUT = os.path.join(ROOT, "natijalar", "v2", "downstream_%s.json" % CORPUS)
os.makedirs(os.path.dirname(OUT), exist_ok=True)

data = fedmf.Data(CORPUS)
res = {"corpus": CORPUS, "n_user": data.n_user, "n_item": data.n_item,
       "n_train": int(data.n_train), "rounds": ROUNDS, "clip": CLIP,
       "delta": DELTA, "lr_item": LR_ITEM, "seeds": SEEDS,
       "accounting": "zCDP, Gaussian rho=C^2/(2 sigma^2) per round, "
                     "eps = rho + 2 sqrt(rho ln(1/delta))",
       "popularity": dict(zip(("recall20", "ndcg20"), fedmf.popularity(data))),
       "nonprivate": {}, "grid": []}

t0 = time.time()
for seed in SEEDS:
    r = fedmf.train(data, rounds=ROUNDS, lr_item=LR_ITEM, clip=CLIP, seed=seed)
    res["nonprivate"].setdefault("full_rank", []).append(
        {"seed": seed, "recall20": r["recall20"], "ndcg20": r["ndcg20"]})
    for rank in RANKS:
        r = fedmf.train(data, rounds=ROUNDS, lr_item=LR_ITEM, clip=CLIP,
                        rank=rank, sigma=0.0, seed=seed)
        res["nonprivate"].setdefault("rank_%d" % rank, []).append(
            {"seed": seed, "recall20": r["recall20"], "ndcg20": r["ndcg20"]})
print("non-private done %.0fs" % (time.time() - t0), flush=True)

for eps in EPS:
    sigma = dp.sigma_for_total_budget(eps, DELTA, ROUNDS, CLIP)
    for rank in RANKS:
        for seed in SEEDS:
            r = fedmf.train(data, rounds=ROUNDS, lr_item=LR_ITEM, clip=CLIP,
                            rank=rank, sigma=sigma, seed=seed)
            res["grid"].append({"eps_total": eps, "sigma": sigma, "rank": rank,
                                "seed": seed, "recall20": r["recall20"],
                                "ndcg20": r["ndcg20"]})
        m = np.mean([g["recall20"] for g in res["grid"]
                     if g["eps_total"] == eps and g["rank"] == rank])
        print("eps=%-5g r=%-3d recall@20=%.4f  (%.0fs)" % (eps, rank, m, time.time() - t0),
              flush=True)
    json.dump(res, open(OUT, "w"), indent=1)

# basis ablation: public random basis vs post-processed release basis
res["basis_ablation"] = []
for eps in (16.0, 64.0):
    sigma = dp.sigma_for_total_budget(eps, DELTA, ROUNDS, CLIP)
    for rank in (4, 16):
        for seed in SEEDS:
            r = fedmf.train(data, rounds=ROUNDS, lr_item=LR_ITEM, clip=CLIP,
                            rank=rank, sigma=sigma, seed=seed, basis="random")
            res["basis_ablation"].append({"eps_total": eps, "rank": rank, "seed": seed,
                                          "basis": "random", "recall20": r["recall20"],
                                          "ndcg20": r["ndcg20"]})
json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT, "%.0fs" % (time.time() - t0))
