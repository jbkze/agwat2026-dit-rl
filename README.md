# DIT-RL revision experiments

Experiments for the major revision of *Benchmarking deep reinforcement learning strategies for
the optimal scheduling of deficit irrigation systems* (Schütze, Kunze, Grové; Agricultural Water
Management, manuscript AGWAT-D-26-00823). All methods schedule maize irrigation in AquaCrop-OSPy
under a hard seasonal water budget: two DRL configurations (DRL1 and the reference DRL2 of
Kelly et al. 2024), simple rules, and an optimizer with perfect weather foresight (the oracle).

The experiment plan, with the reviewer comment each experiment answers, is in
[`docs/experiment-plan.md`](docs/experiment-plan.md). Design decisions are in
[`docs/decisions.md`](docs/decisions.md).

## Setup

```bash
make install          # uv venv with Python 3.10 + all pinned dependencies (CPU torch)
make smoke            # all code paths end to end in a few minutes
make test             # minimal checks of budget handling and statistics
```

Training runs on CPU. The GPU does not help: the crop model is sequential Python, and one job
uses one core (threads are pinned). On the 64-core h200 server, `WORKERS=60` is a good value.

## Running an experiment

```bash
make dry-run CONFIG=config/experiments/e2_ablation.yaml     # list the jobs
make run     CONFIG=config/experiments/e2_ablation.yaml WORKERS=60
# after an interruption:
.venv/bin/python scripts/run.py config/experiments/e2_ablation.yaml \
    --resume results/runs/2026-09-30_e2_ablation
```

Each run writes `results/runs/<date>_<experiment>/` with a copy of all configs, a manifest (git
commit, package versions, machine), a log, and one folder per job with its log, results, the
trained model and the training curves. See the docstring of `src/ditrl/runner.py`.

| Config | Experiment | Jobs |
|---|---|---|
| `check_paper.yaml`, `check_oracle.yaml` | comparison with the published Tunis values | 12, 2 |
| `e0_oracle_ds.yaml`, `e0_oracle_ss.yaml` | E0: perfect-foresight and robust open-loop schedules | 927, 45 |
| `e1_benchmark.yaml` | E1: all methods at all budgets | 354 |
| `e2_ablation.yaml` | E2: 2 x 2 x 2 ablation | 640 |
| `e3a_forecast.yaml`, `e3b_test_year.yaml` | E3: value of weather information | 80, 320 |
| `e4_low_budget.yaml` | E4: failure at 100–200 mm | 160 |
| `e5_robustness.yaml` | E5: noisy observations (uses the models of E1 and E2) | per model |

## From runs to the paper

1. Enter the run folders in `config/analysis.yaml`.
2. `make collect` builds flat tables in `results/derived/`.
3. `make paper-assets` regenerates all figures, tables and `macros.tex` (registry:
   `analysis/assets.py`) and copies them into `generated/` of the paper repo
   (path in `config/paths.local.yaml`, template in `config/paths.local.example.yaml`).
4. `make check-drift` before submission: does the paper still match the results?

**Oracle source.** Until the GET-OPTIS code is available, the oracle values come from the pymoo
runs of E0. To switch, put the GET-OPTIS results into `data/external/` (format in
`data/README.md`) and set `oracle.source: getoptis` in `config/analysis.yaml`.

## Layout

```
config/          sites, methods, experiments, analysis (all parameters live here)
src/ditrl/       sites.py (AquaCrop setup), env.py, observations.py, drl.py, heuristics.py,
                 oracle.py, config.py (job expansion), jobs.py, runner.py
scripts/run.py   command line entry point
analysis/        collect.py, stats.py, fig_*.py, tables.py, macros.py, assets.py
data/            Nebraska weather; external results
results/         runs/ (immutable, not in git), derived/ (regenerable)
docs/            experiment plan, decision log
```
