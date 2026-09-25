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
make stage   STAGE=1 WORKERS=60                              # a whole stage, resumable
make dry-run CONFIG=config/experiments/s1_ablation.yaml     # list the jobs of one experiment
make run     CONFIG=config/experiments/s1_ablation.yaml WORKERS=60
# one experiment after an interruption:
.venv/bin/python scripts/run.py config/experiments/s1_ablation.yaml \
    --resume results/runs/2026-09-30_s1_ablation
```

Each run writes `results/runs/<date>_<experiment>/` with a copy of all configs, a manifest (git
commit, package versions, machine), a log, and one folder per job with its log, results, the
trained model and the training curves. See the docstring of `src/ditrl/runner.py`.

The experiments are grouped in three stages (`docs/experiment-plan.md`): stage 1 is the
minimum for the revision, stages 2 and 3 add to it. `make stage STAGE=1` runs stage 1;
`STAGE=2` runs everything of stages 1 and 2 that is still missing, and so on.

| Stage | Configs (`config/experiments/`) | DRL trainings |
|---|---|---|
| 1 | `s1_benchmark`, `s1_ablation` | 200 |
| 2 | `s2_benchmark_seeds`, `s2_ablation_seeds`, `s2_ablation_interactions`, `s2_benchmark_budgets`, `s2_low_budget`, `s2_robustness` | 780 |
| 3 | `s3_forecast`, `s3_test_year`, `s3_obs_scaled`, `s3_oracle_ds`, `s3_oracle_ss` | 480 + oracle |

Checks and diagnoses that are already done: `check_*` (comparison with the published Tunis
values) and `diag_drl2_*` (cause of the weak published DRL2 values, `docs/diagnosis-drl2.md`).

## From runs to the paper

1. `config/analysis.yaml` lists which runs feed each derived table; runs that do not exist yet
   are skipped, so the analysis works after every stage.
2. `make collect` builds flat tables in `results/derived/`.
3. `make stage-assets` regenerates all figures, tables and `macros.tex` into `paper_assets/`
   (registry: `analysis/assets.py`); assets whose inputs are missing are skipped.
4. `make paper-assets` does the same and copies them into `generated/` of the paper repo (path in
   `config/paths.local.yaml`); `make check-drift` before submission.

**Oracle source.** The oracle values come from `data/external/getoptis_ea_*.csv`, which hold the
published GET-OPTIS values of the manuscript tables. New GET-OPTIS results replace these files;
the own pymoo oracle of stage 3 is used with `oracle.source: pymoo` in `config/analysis.yaml`.

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
