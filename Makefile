.PHONY: install run all dry-run smoke test collect stage-assets paper-assets check-drift

# Windows default; on Linux call make PY=.venv/bin/python ...
PY ?= .venv/Scripts/python
WORKERS ?= 60

install:
	uv venv --python 3.10 .venv
	uv pip install --python $(PY) -e . --index-strategy unsafe-best-match \
		--extra-index-url https://download.pytorch.org/whl/cpu

# make run CONFIG=config/experiments/e2_ablation.yaml [WORKERS=60]
run:
	$(PY) scripts/run.py $(CONFIG) --workers $(WORKERS)

# Every experiment in order, resumable, then derived tables and paper assets
all:
	PYTHONPATH=. $(PY) scripts/run_all.py --workers $(WORKERS)

dry-run:
	$(PY) scripts/run.py $(CONFIG) --dry-run

# End-to-end check of all code paths in a few minutes; writes results/runs/<date>_smoke*
smoke:
	$(PY) scripts/run.py config/experiments/smoke.yaml --workers 2 --tag _$(shell date +%H%M%S)
	$(PY) scripts/run.py config/experiments/smoke_oracle.yaml --workers 1 --tag _$(shell date +%H%M%S)

test:
	PYTHONPATH=. $(PY) tests/test_core.py

collect:
	$(PY) -m analysis.collect

stage-assets:
	PYTHONPATH=. $(PY) -m analysis.assets stage

paper-assets:
	$(PY) -m analysis.assets export

check-drift:
	$(PY) -m analysis.assets check
