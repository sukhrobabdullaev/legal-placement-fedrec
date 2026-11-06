# -*- coding: utf-8 -*-
"""Revised Section 6.1: placement at measured demands under (i) a legality
mask that follows the statute text entry by entry, (ii) several external
providers and price types, (iii) a user source/sink in Uzbekistan, and
(iv) a two-dimensional break-even map. Also locates the data-volume scale at
which inter-layer transfer, rather than compute price, starts to decide the
plan, i.e. where the dynamic programme does more than a per-stage argmin.

Prices (USD per vCPU-hour, memory included at the stated GB per vCPU) are
listed in PRICES with their sources; all were read on 5 November 2026 unless
stated. Run from real/:  python run_joylashtirish_v2.py
"""
import json, os, sys, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from adaptreal2.narx import joylashtir, L0_VCPU_US_KOLO, L0_VCPU_UZ_KOLO
from adaptreal2.bosqichlar import NOMLAR

EURUSD = 1.1267          # ECB reference rate, 4 Nov 2026
USD_UZS = 11847.16       # CBU rate used in the original price snapshot (21 Aug 2026)

# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------
L1 = {
    # UzCloud public list (23 Aug 2026): 118 UZS/vCPU-h + 17 UZS/GB-h
    "UzCloud, 4 GB/vCPU": (118 + 4 * 17) / USD_UZS,
    "UzCloud, 2 GB/vCPU": (118 + 2 * 17) / USD_UZS,
}
L2 = {
    # name: (USD per vCPU-hour, GB per vCPU, vCPU type, egress USD/GB, source)
    "AWS m7i.large on-demand": (0.1207 / 2, 4, "hyperthread", 0.09,
                                "aws-pricing.com, updated 2026-10-31"),
    "AWS m7i.large RI 1-yr": (58.31 / 730 / 2, 4, "hyperthread", 0.09,
                              "aws-pricing.com, updated 2026-10-31"),
    "AWS m7i.large RI 3-yr": (39.98 / 730 / 2, 4, "hyperthread", 0.09,
                              "aws-pricing.com, updated 2026-10-31"),
    "AWS m7i.large spot": (0.0557 / 2, 4, "hyperthread", 0.09,
                           "aws-pricing.com, updated 2026-10-31"),
    "GCP e2-standard-2 on-demand": (0.0863 / 2, 4, "shared", 0.12,
                                    "gcloud-compute.com, updated 2026-11-01"),
    "Hetzner CCX13 (dedicated)": (0.0697 * EURUSD / 2, 4, "dedicated", 1.0 * EURUSD / 1000,
                                  "hetzner.com/cloud/general-purpose, 2026-11-05"),
    "Hetzner CX23 (shared)": (0.0096 * EURUSD / 2, 2, "shared", 1.0 * EURUSD / 1000,
                              "hetzner.com/cloud/cost-optimized, 2026-11-05"),
}
L0 = {"on-prem, UZ colocation": L0_VCPU_UZ_KOLO,
      "on-prem, US colocation": L0_VCPU_US_KOLO}


def uzatish(l2_egress, yuklama=0.30):
    """Inter-layer price, USD/GB. L0<->L1 over Tas-IX is free; leaving the
    country from L0/L1 uses the committed-Mbps uplink (original model);
    leaving L2 uses that provider's internet egress."""
    out_uz = 0.0104 / yuklama
    return {(0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0, (0, 1): 0.0, (1, 0): 0.0,
            (0, 2): out_uz, (1, 2): out_uz, (2, 0): l2_egress, (2, 1): l2_egress}


# ---------------------------------------------------------------------------
# legal masks, one row per (stage, regime) with the provision it rests on
# ---------------------------------------------------------------------------
# Data handled by each stage. The pipeline serves an e-commerce marketplace
# that is not a telecom operator and processes no biometric or genetic data.
STAGE_DATA = {
    "ingestion": "user id, item id, timestamp (personal data)",
    "feature": "user history and state (personal data)",
    "retrieval": "user id and history-derived scores (personal data)",
    "pre_ranking": "user history, candidate scores (personal data)",
    "ranking": "user history, fused scores (personal data)",
    "re_ranking": "user history, candidate titles (personal data)",
    "logging": "decision record with user id (personal data)",
    "analytics": "daily aggregates without identifiers (not personal data)",
}
PERSONAL = {n: n != "analytics" for n in NOMLAR}

REGIMES = {
    "A. no restriction": {
        "rule": "counterfactual baseline",
        "allow": lambda n: {0, 1, 2}},
    "B. 2021 Art. 27-1 (broad reading)": {
        "rule": "ZRU-547 Art. 27-1 as introduced by ZRU-666 (2021): personal data of "
                "citizens collected, systematised and stored on technical means in Uzbekistan",
        "allow": lambda n: {0, 1} if PERSONAL[n] else {0, 1, 2}},
    "C1. 2026, EU destination on adequacy list": {
        "rule": "Art. 27-1 as replaced by ZRU-1125 (2026): mandatory domestic storage only for "
                "biometric, genetic and telecom-subscriber data; other personal data abroad if the "
                "destination is on the adequacy list (Germany listed, CoM resolution of July 2026)",
        "allow": lambda n: {0, 1, 2}},
    "C2. 2026, non-listed destination, no SCC/BCR": {
        "rule": "ZRU-1125: none of the three alternative conditions met -> personal data stay domestic",
        "allow": lambda n: {0, 1} if PERSONAL[n] else {0, 1, 2}},
    "C3. 2026, operator is a telecom (subscriber data)": {
        "rule": "ZRU-1125: data of subscribers of telecom operators active in Uzbekistan -> domestic",
        "allow": lambda n: {0, 1} if PERSONAL[n] else {0, 1, 2}},
}


def mask(reg):
    return [[l in REGIMES[reg]["allow"](n) for l in (0, 1, 2)] for n in NOMLAR]


def load_profile(path):
    p = json.load(open(path))["bosqichlar"]
    cpu = [p[n]["cpu_soat_1M_sorov"] for n in NOMLAR]
    gb = [p[n]["gb_chiqish_1M_sorov"] for n in NOMLAR]
    resp_gb = p["re_ranking"]["bayt_chiqish_sorov"] * 1e6 / 1024 ** 3
    in_gb = p["ingestion"]["bayt_kirish_sorov"] * 1e6 / 1024 ** 3
    return cpu, gb, resp_gb, in_gb


def solve(cpu, gb, vcpu, uz, msk, resp_gb, in_gb):
    """Placement with a user source and sink in Uzbekistan.
    The request enters ingestion from Uzbekistan and the ranked list leaves
    re-ranking for Uzbekistan; both legs are added to the per-stage cost of
    the external layer, which keeps the chain recursion exact."""
    extra = np.zeros((len(NOMLAR), 3))
    extra[NOMLAR.index("ingestion"), 2] += in_gb * uz[(1, 2)]
    extra[NOMLAR.index("re_ranking"), 2] += resp_gb * uz[(2, 1)]
    N = len(cpu); INF = float("inf")
    dp_ = np.full((N, 3), INF); arg = np.full((N, 3), -1)
    for l in range(3):
        if msk[0][l]:
            dp_[0, l] = cpu[0] * vcpu[l] + extra[0, l]
    for i in range(1, N):
        for l in range(3):
            if not msk[i][l]:
                continue
            c = [dp_[i - 1, lp] + gb[i - 1] * uz[(lp, l)] for lp in range(3)]
            lp = int(np.argmin(c))
            if np.isfinite(c[lp]):
                dp_[i, l] = c[lp] + cpu[i] * vcpu[l] + extra[i, l]; arg[i, l] = lp
    l = int(np.argmin(dp_[-1])); cost = float(dp_[-1, l])
    plan = [l]
    for i in range(N - 1, 0, -1):
        l = int(arg[i, l]); plan.append(l)
    return cost, plan[::-1]


def main():
    prof = {"measured (x86-64)": os.path.join(HERE, "natijalar", "bosqich_profili.json"),
            "re-measured (arm64)": os.path.join(HERE, "natijalar", "qayta_2026-11-05",
                                                "arm64", "bosqich_profili.json")}
    out = {"prices": {"L0": L0, "L1": L1,
                      "L2": {k: {"usd_per_vcpu_h": v[0], "gb_per_vcpu": v[1],
                                 "vcpu": v[2], "egress_usd_gb": v[3], "source": v[4]}
                             for k, v in L2.items()},
                      "eurusd": EURUSD},
           "audit": [{"stage": n, "data": STAGE_DATA[n], "regime": r,
                      "allowed_layers": sorted(REGIMES[r]["allow"](n)),
                      "provision": REGIMES[r]["rule"]} for r in REGIMES for n in NOMLAR],
           "scenarios": []}
    for pname, ppath in prof.items():
        cpu, gb, resp_gb, in_gb = load_profile(ppath)
        for l0n, l1n, l2n, yuk in itertools.product(L0, L1, L2, (0.2, 0.3, 1.0)):
            v = {0: L0[l0n], 1: L1[l1n], 2: L2[l2n][0]}
            uz = uzatish(L2[l2n][3], yuk)
            row = {"profile": pname, "L0": l0n, "L1": l1n, "L2": l2n, "uplink_util": yuk,
                   "ratio_L2_L1": v[2] / v[1], "ratio_L0_L1": v[0] / v[1], "regimes": {}}
            base = None
            for reg in REGIMES:
                c, plan = solve(cpu, gb, v, uz, mask(reg), resp_gb, in_gb)
                base = c if base is None else base
                row["regimes"][reg] = {"usd_1M": c, "plan": plan,
                                       "premium_pct": (c / base - 1) * 100}
            out["scenarios"].append(row)

    # headline table: x86 profile, UZ colocation, UzCloud 4 GB/vCPU, 30% uplink
    head = [s for s in out["scenarios"] if s["profile"] == "measured (x86-64)"
            and s["L0"] == "on-prem, UZ colocation" and s["L1"] == "UzCloud, 4 GB/vCPU"
            and s["uplink_util"] == 0.3]
    print("%-30s %6s | %s" % ("external provider", "L2/L1", " | ".join(r[:3] for r in REGIMES)))
    for s in head:
        print("%-30s %6.3f | %s" % (s["L2"], s["ratio_L2_L1"], " | ".join(
            "%.3f %+5.1f%%" % (s["regimes"][r]["usd_1M"], s["regimes"][r]["premium_pct"])
            for r in REGIMES)))

    # 2-D break-even map: premium of regime B over A
    cpu, gb, resp_gb, in_gb = load_profile(prof["measured (x86-64)"])
    l1 = L1["UzCloud, 4 GB/vCPU"]
    xs = np.round(np.geomspace(0.1, 4.0, 41), 4)       # L2/L1
    ys = np.round(np.geomspace(0.25, 2.0, 31), 4)      # L0/L1
    maps = {}
    for tag, egress in (("egress_0.0011", L2["Hetzner CX23 (shared)"][3]), ("egress_0.09", 0.09)):
        Z = np.zeros((len(ys), len(xs)))
        uz = uzatish(egress, 0.3)
        for a, y in enumerate(ys):
            for b, x in enumerate(xs):
                v = {0: y * l1, 1: l1, 2: x * l1}
                ca, _ = solve(cpu, gb, v, uz, mask("A. no restriction"), resp_gb, in_gb)
                cb, _ = solve(cpu, gb, v, uz, mask("B. 2021 Art. 27-1 (broad reading)"), resp_gb, in_gb)
                Z[a, b] = (cb / ca - 1) * 100
        maps[tag] = Z.round(3).tolist()
    out["breakeven_map"] = {"L2_over_L1": xs.tolist(), "L0_over_L1": ys.tolist(),
                            "premium_pct": maps}
    Z = np.array(maps["egress_0.0011"])
    # analytic boundary check: premium > 0 iff L2 < min(L0, L1) (up to transfer)
    viol = 0
    for a, y in enumerate(ys):
        for b, x in enumerate(xs):
            pred = x < min(y, 1.0) * (1 - 1e-3)
            viol += int(pred != (Z[a, b] > 1e-6))
    out["breakeven_rule_mismatches"] = viol

    # when does transfer decide? scale inter-stage volumes by kappa
    v = {0: L0["on-prem, UZ colocation"], 1: l1, 2: L2["Hetzner CX23 (shared)"][0]}
    uz = uzatish(L2["Hetzner CX23 (shared)"][3], 0.3)
    kap = []
    for k in np.geomspace(1, 1e5, 51):
        c, plan = solve(cpu, list(np.array(gb) * k), v, uz, mask("A. no restriction"),
                        resp_gb * k, in_gb * k)
        kap.append({"kappa": float(k), "usd_1M": c, "plan": plan})
    out["volume_scaling_hetzner"] = kap
    first_mixed = next((r for r in kap if len(set(r["plan"])) > 1), None)
    out["first_mixed_plan"] = first_mixed      # first scanned kappa whose plan uses >1 layer
    # exact break-evens (bisection) for both profiles, both on-premises prices and two egress prices
    def breakeven(cpu_, gb_, rgb_, igb_, l0, eg):
        uz_ = uzatish(eg, 0.3); lo, hi = 0.01, 3.0
        for _ in range(60):
            x = (lo + hi) / 2; v_ = {0: l0, 1: l1, 2: x * l1}
            a_, _ = solve(cpu_, gb_, v_, uz_, mask("A. no restriction"), rgb_, igb_)
            b_, _ = solve(cpu_, gb_, v_, uz_, mask("B. 2021 Art. 27-1 (broad reading)"), rgb_, igb_)
            lo, hi = (x, hi) if b_ / a_ - 1 > 1e-9 else (lo, x)
        return hi
    out["breakeven_exact"] = []
    for pname, ppath in prof.items():
        c_, g_, r_, i_ = load_profile(ppath)
        for l0n, l0v in L0.items():
            for eg in (L2["Hetzner CX23 (shared)"][3], 0.09):
                out["breakeven_exact"].append({"profile": pname, "L0": l0n, "L0_over_L1": l0v / l1,
                                               "egress_usd_gb": eg,
                                               "L2_over_L1": breakeven(c_, g_, r_, i_, l0v, eg)})
    # exact volume threshold at which the shared-vCPU plan returns home
    kap_fine, plans = None, set()
    for k in np.geomspace(1, 200, 2000):
        c, plan = solve(cpu, list(np.array(gb) * k), v, uz, mask("A. no restriction"),
                        resp_gb * k, in_gb * k)
        plans.add(tuple(plan))
        if kap_fine is None and all(x == 0 for x in plan):
            kap_fine = float(k)
    out["volume_threshold_kappa"] = kap_fine
    out["volume_distinct_plans"] = [list(p_) for p_ in sorted(plans)]
    # plans that differ between the two stage profiles
    idx = {}
    for s_ in out["scenarios"]:
        idx.setdefault((s_["L0"], s_["L1"], s_["L2"], s_["uplink_util"]), {})[s_["profile"]] = s_
    diffs, ratios = [], []
    for key_, d_ in idx.items():
        a_, b_ = d_["measured (x86-64)"], d_["re-measured (arm64)"]
        for reg in REGIMES:
            ratios.append(b_["regimes"][reg]["usd_1M"] / a_["regimes"][reg]["usd_1M"])
            if a_["regimes"][reg]["plan"] != b_["regimes"][reg]["plan"]:
                diffs.append({"L0": key_[0], "L1": key_[1], "L2": key_[2], "uplink_util": key_[3],
                              "regime": reg, "plan_x86": a_["regimes"][reg]["plan"],
                              "plan_arm64": b_["regimes"][reg]["plan"]})
    out["profile_plan_differences"] = diffs
    out["profile_cost_ratio_range"] = [min(ratios), max(ratios)]
    print("break-even rule mismatches on grid:", viol)
    print("kappa:", kap_fine, "plan diffs between profiles:", len(diffs), "cost ratio", min(ratios), max(ratios))
    print("first mixed (non-trivial) plan at volume x", first_mixed and round(first_mixed["kappa"], 1),
          first_mixed and first_mixed["plan"])
    path = os.path.join(HERE, "natijalar", "v2", "placement_v2.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
