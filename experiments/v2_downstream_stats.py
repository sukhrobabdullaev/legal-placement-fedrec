# -*- coding: utf-8 -*-
"""Summary statistics of the downstream grid (all seeds): mean and s.d. per
cell, the best rank per budget, the ranks statistically tied with it
(Welch t-test, p >= 0.05), and the Welch test of the best low rank (r <= 2)
against full rank (r = 64) per budget. Writes natijalar/v2/downstream_stats.json.
Run from the package root after v2_downstream_dp.py (and v2_downstream_extra.py)."""
import json, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "natijalar", "v2", "downstream_ml-1m.json")))
eps_list = sorted({g["eps_total"] for g in D["grid"]})
ranks = sorted({g["rank"] for g in D["grid"]})
cell = lambda e, r, k="recall20": np.array([g[k] for g in D["grid"] if g["eps_total"] == e and g["rank"] == r])
out = {"seeds": D["seeds"], "n_seeds": len(D["seeds"]), "rows": []}
for e in eps_list:
    m = {r: float(cell(e, r).mean()) for r in ranks}
    sd = {r: float(cell(e, r).std(ddof=1)) for r in ranks}
    best = max(m, key=m.get)
    tied = [r for r in ranks if r == best or stats.ttest_ind(cell(e, best), cell(e, r), equal_var=False).pvalue >= 0.05]
    low = max((1, 2), key=lambda r: m[r])
    t = stats.ttest_ind(cell(e, low), cell(e, 64), equal_var=False)
    out["rows"].append({"eps_total": e, "mean": m, "sd": sd, "best_rank": best, "tied_with_best": tied,
                        "best_low_rank": low, "low_minus_full": m[low] - m[64],
                        "welch_low_vs_full_p": float(t.pvalue),
                        "ndcg_mean": {r: float(cell(e, r, "ndcg20").mean()) for r in ranks}})
    print(e, "best", best, "tied", tied, "low-full %.4f p=%.2g" % (m[low] - m[64], t.pvalue))
np_full = [x["recall20"] for x in D["nonprivate"]["full_rank"]]
out["nonprivate_full_mean"] = float(np.mean(np_full)); out["nonprivate_full_sd"] = float(np.std(np_full, ddof=1))
out["popularity_recall20"] = D["popularity"]["recall20"]
json.dump(out, open(os.path.join(ROOT, "natijalar", "v2", "downstream_stats.json"), "w"), indent=1)
