# -*- coding: utf-8 -*-
"""Revised Section 6.5: composing the two mechanisms on one worked example.

Architecture: clients in Uzbekistan send clipped updates to an aggregator in
a domestic layer, which adds the Gaussian noise and releases A_t = G_t V_t + N
(d1 x r numbers per round; V_t is public). Only A_t may cross the border to a
model-training service in the external layer. Whether it may cross is a
legality-mask question about the legal class of a DP release, which the
statute does not settle; both readings are encoded:
    "derived personal data"  -> class follows the personal-data rule of the regime
    "anonymised output"      -> class OPEN
For each regime and reading the script reports whether the external
aggregator is lawful, and, from the downstream grid, the utility-optimal rank,
its payload and its cost at the measured uplink price.

Run from the package root after v2_downstream_dp.py.
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "natijalar", "v2", "downstream_ml-1m.json")))
D1, FULL, BYTES = D["n_item"], 64, 4
UPLINK_USD_GB = 0.0104 / 0.30           # domestic international uplink, 30% utilisation

personal_abroad = {"A. no restriction": True,
                   "B. 2021 Art. 27-1 (broad reading)": False,
                   "C1. 2026, EU destination on adequacy list": True,
                   "C2. 2026, non-listed destination, no SCC/BCR": False,
                   "C3. 2026, operator is a telecom (subscriber data)": False}
legal = []
for reg, ok in personal_abroad.items():
    legal.append({"regime": reg,
                  "release_as_personal_data": ok,
                  "release_as_anonymised": True})

rows = []
eps_list = sorted({g["eps_total"] for g in D["grid"]})
ranks = sorted({g["rank"] for g in D["grid"]})
for e in eps_list:
    m = {r: np.mean([g["recall20"] for g in D["grid"] if g["eps_total"] == e and g["rank"] == r])
         for r in ranks}
    rb = max(m, key=m.get)
    pay = D1 * rb * BYTES * D["rounds"] / 1e6
    rows.append({"eps_total": e, "best_rank": rb, "recall20_best": m[rb],
                 "recall20_full_rank": m[FULL],
                 "payload_MB_per_run_best": pay,
                 "payload_MB_per_run_full": D1 * FULL * BYTES * D["rounds"] / 1e6,
                 "payload_reduction_x": FULL / rb,
                 "uplink_cost_usd_per_run_best": pay / 1000 * UPLINK_USD_GB})
out = {"legal": legal, "schedule": rows, "uplink_usd_per_gb": UPLINK_USD_GB}
json.dump(out, open(os.path.join(ROOT, "natijalar", "v2", "composition.json"), "w"), indent=1)
for r in rows:
    print("eps=%-4g best r=%-2d recall %.4f (full rank %.4f)  payload %.1f MB/run (x%.0f smaller)  "
          "cost %.5f USD" % (r["eps_total"], r["best_rank"], r["recall20_best"], r["recall20_full_rank"],
                             r["payload_MB_per_run_best"], r["payload_reduction_x"],
                             r["uplink_cost_usd_per_run_best"]))
