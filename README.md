# legal-placement-fedrec

Reproducibility package for the article

> **Legality-Aware Placement of Recommender Pipelines and Private Low-Rank Update Streams under Data-Localization Law**
> Sukhrobjon Abdullaev, Tashkent University of Information Technologies named after Muhammad al-Khwarizmi.

The package regenerates every number, table and figure in the article.

| Part | Article sections | What it does |
|---|---|---|
| `adaptrec/` + `experiments/` | 4, 5, 6.2 (synthetic) | Legality masks and exact O(N\|L\|²) placement by dynamic programming; the privacy–compression–utility rank law r\* ∝ ε^{2/(ζ+1)} on synthetic power-law spectra (ζ = 0.5, 1, 2). |
| `real/` | 6.1, 6.2 (measured) | Per-stage CPU and data-volume demands measured on a working recommender pipeline; placement at published prices (UzCloud, AWS Frankfurt, CBU exchange rate, accessed 23 August 2026) under four legal regimes of Uzbekistan's Law ZRU-547; break-even sweep; spectrum of real federated update matrices (BPR-MF on Amazon Reviews 2023) and the rank law on it. |

Code comments and some identifiers are in Uzbek (`natijalar` = results, `bosqich` = stage, `narx` = price, `joylashtirish` = placement, `qayta` = re-run). An Uzbek README is in `README_uz.md`.

## Install and data

```bash
python -m pip install -r requirements.txt          # Python 3.10+
git clone https://github.com/sukhrobabdullaev/adaptrec-routing   # companion article; its real/ pipeline is reused
export MAQOLA1_REAL=/path/to/adaptrec-routing/real
export VG_DATA=/path/to/vg       # vg_stream.csv + vg_items.csv, built as described in adaptrec-routing/README.md
export MINILM=/path/to/minilm    # only for run_bosqich.py (cross-encoder stage)
```

Data: Amazon Reviews 2023 (McAuley Lab), category `Video_Games`, 5-core: 857,505 events, 98,906 users, 26,354 items. No data are redistributed here.

## Run

```bash
make sintetik     # placement scenarios + synthetic rank law (about 20 s)
make real         # stage profile, placement, break-even, spectrum, rank law, Figures 1 and 3 (about 5 min)
```

`real/run_sindirish.py` was added in this release: it regenerates the break-even sweep, the measured price ratio and the cross-encoder cost effect that the original run wrote into `real/natijalar/joylashtirish.json`. Its output matches the stored file exactly.

## Independent re-run (5 November 2026)

All experiments were regenerated from the raw `.jsonl.gz` files. Outputs are in `real/natijalar/qayta_2026-11-05/`.

| Result | Original | x86-64 re-run | arm64 re-run |
|---|---|---|---|
| Synthetic rank law (slopes 1.40 / 1.02 / 0.68 vs 1.33 / 1.00 / 0.67) | | not run | byte-identical |
| Placement scenarios, synthetic | | not run | byte-identical |
| Placement at measured prices: 0.264 USD per 1M requests, all four regimes on-premises, compliance cost 0%; break-even 0.45 to 0.5; +9.1%, +22.7%, +390.8%; cross-encoder accounts for 90.7% of cost | | not run | identical (same stored stage profile) |
| Spectrum of real update matrices: ζ = 0.037, power-law R² = 0.904, exponential R² = 0.635 | | identical to 1e-15 | ζ = 0.048, R² = 0.907 / 0.644 |
| Rank law on real spectrum: slope 2.017 vs predicted 1.928 (4.6%) | | byte-identical | 2.014 vs 1.908 (5.5%), R² = 0.950 |
| Downstream variant (BPR-MF NDCG@10 = 0.0034) | | identical | not run |

Two quantities are machine dependent by nature:

* **Stage CPU time** is a wall-clock measurement. Re-measured on the arm64 machine it is 60.4 instead of 34.3 CPU-hours per million requests (almost entirely the cross-encoder stage). Data volumes are unchanged. With the re-measured profile the optimal plan, the zero compliance cost and the break-even ratio are unchanged (`arm64/joylashtirish_qayta_profil.json`).
* **BPR-MF training** is stochastic gradient descent in floating point; on x86-64 it reproduces the original exactly, on arm64 BLAS rounding differences accumulate over 40 epochs and give ζ = 0.048. The rank law holds on both.

## Revision 2 (November 2026)

The revised article replaces three parts of the first version. Outputs are in `natijalar/v2/` and `real/natijalar/v2/`; the figures used in the article are `figures/v3_*.{png,pdf}`.

| What changed | Why | Script |
|---|---|---|
| Private release: the projection basis is fixed before each round (public seed or post-processing of the previous release); zCDP accounting over all rounds; analytic Gaussian for single releases | v1 took `V_r` from the SVD of the private aggregate and sent it without noise, so the release was not differentially private; the classical Gaussian bound is invalid for eps > 1 | `adaptrec/dp.py`, `experiments/v2_mechanism.py` |
| Optimal rank stated exactly: keep direction j while its energy exceeds d1*sigma^2 (Proposition 1); the power law is only a corollary | the v1 "law" was this threshold rule in disguise; the Monte Carlo reproduced it at all 14 points | `adaptrec/dp.py` |
| Spectra of federated update matrices on four public corpora (MovieLens-1M, Gowalla, Yelp2018, Amazon-Book), five rounds each, 50 independent draws per round | v1 used one matrix from one round of one corpus | `experiments/v2_spectra.py` |
| Private federated BPR-MF on MovieLens-1M: 6 total budgets x 7 ranks x 3 seeds, by Recall@20/NDCG@20 | the rank trade-off is now measured by recommendation quality under an end-to-end private mechanism | `adaptrec/fedmf.py`, `experiments/v2_downstream_dp.py` |
| Placement: statute-traceable masks (2021, 2026 with and without an adequacy-listed destination, telecom operator), seven external price points, user source and sink, two-dimensional break-even map, transfer-volume threshold | the v1 2026 mask kept e-commerce data LOCAL although the 2026 text does not; the single break-even was the assumed on-premises price ratio | `real/run_joylashtirish_v2.py`, `real/make_fig_v2_placement.py` |
| Final article figures (Figures 1-3 of the revised article, PNG and PDF) | single consistent style for submission; `figures/v2_*.png` are the intermediate versions | `experiments/make_figures_v3.py` -> `figures/v3_*.{png,pdf}` |
| Composition worked example | links the rank schedule to the border payload | `experiments/v2_composition.py` |
| Five seeds per downstream cell, Welch tests, adaptive-rank comparison | the best rank is reported with the ranks statistically tied with it | `experiments/v2_downstream_extra.py`, `v2_downstream_stats.py`, `v2_auto_rank.py` |
| Seeds and clipping sensitivity of the spectra; v1 reference quantities | spectra are not an artifact of one run or of clipping | `SEED`, `CLIP`, `TAG` variables of `v2_spectra.py`; `v2_v1_reference.py` |

Data for the update-stream experiments: standard splits from https://github.com/hexiangnan/neural_collaborative_filtering (MovieLens-1M, commit 4aab159) and https://github.com/gusye1234/LightGCN-PyTorch (Gowalla, Yelp2018, Amazon-Book, commit 947ca2b). Put `ml-1m.{train,test}.rating` and `<name>.{train,test}.txt` in `$REC_DATA` (default `~/data`).

```bash
make v2      # about 3 hours on 2 CPU cores
```

`natijalar/v2/old_bootstrap/` keeps the first spectra run with a client bootstrap, which biased the intervals; it is kept only to document why the bootstrap was replaced.

## Layout

```
adaptrec/           legality masks, exact placement, privacy mechanism (synthetic part)
experiments/        placement_scenarios.py, rank_law.py
real/adaptreal2/    stage accounting, price model and DP placement
real/run_*.py       measured experiments
real/natijalar/     original outputs (August 2026) and qayta_2026-11-05/ re-run outputs
figures/            Figures 1-3 of the article
```

## Citation

See `CITATION.cff`. License: MIT.
