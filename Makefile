PY ?= python3

.PHONY: all sintetik real v2 clean
all: sintetik real

# sintetik rang qonuni
sintetik:
	$(PY) experiments/placement_scenarios.py
	$(PY) experiments/rank_law.py

# 6.1-6.2: o'lchangan profillar, real narxlar, real spektr
real:
	cd real && $(PY) run_bosqich.py
	cd real && $(PY) run_joylashtirish.py
	cd real && $(PY) run_sindirish.py
	cd real && $(PY) run_spektr.py
	cd real && $(PY) run_zeta2.py
	cd real && $(PY) make_fig1.py
	cd real && $(PY) make_fig3.py

clean:
	rm -rf natijalar figures real/natijalar real/figures

# 2-tahrir (noyabr 2026): to'g'ri DP mexanizmi, ko'p korpusli spektr, downstream DP, placement v2
v2:
	$(PY) experiments/v2_mechanism.py
	$(PY) experiments/v2_downstream_dp.py
	$(PY) experiments/v2_downstream_extra.py
	$(PY) experiments/v2_downstream_stats.py
	$(PY) experiments/v2_auto_rank.py
	$(PY) experiments/v2_spectra.py
	SEED=1 TAG=_seed1 $(PY) experiments/v2_spectra.py ml-1m yelp2018
	SEED=2 TAG=_seed2 $(PY) experiments/v2_spectra.py ml-1m
	CLIP=10 TAG=_clip10 $(PY) experiments/v2_spectra.py ml-1m
	CLIP=1e9 TAG=_noclip $(PY) experiments/v2_spectra.py ml-1m
	$(PY) experiments/v2_v1_reference.py
	$(PY) experiments/v2_composition.py
	$(PY) experiments/make_fig_v2_rank.py
	cd real && $(PY) run_joylashtirish_v2.py
	cd real && $(PY) make_fig_v2_placement.py
	$(PY) experiments/make_figures_v3.py
