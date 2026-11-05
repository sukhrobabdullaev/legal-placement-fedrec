PY ?= python3

.PHONY: all sintetik real clean
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
