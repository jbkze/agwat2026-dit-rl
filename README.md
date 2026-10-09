# DIT-RL revision experiments

Experiments for the major revision of *Benchmarking deep reinforcement learning strategies for the optimal scheduling of deficit irrigation systems* (Schütze, Kunze, Grové; Agricultural Water Management, manuscript AGWAT-D-26-00823). All methods schedule maize irrigation in AquaCrop-OSPy under a hard seasonal water budget: two DRL configurations (DRL1 and the reference DRL2 of Kelly et al. 2024), simple rules, and an optimizer with perfect weather foresight (the oracle).

## Where things are

| Path | Content |
|---|---|
| `config/plan.yaml` | the whole experiment plan: sites, methods, experiments by stage, analysis settings. Every number lives here |
| `docs/experiment-plan.md` | why each experiment exists and which reviewer comment it answers |
| `docs/decisions.md` | design decisions with date and reason |
| `src/ditrl/` | the experiment code (below) |
| `src/ditrl/analysis/` | tables, statistics, figures and LaTeX macros for the paper |
| `data/` | Nebraska weather and external results (`data/README.md`) |
| `results/` | job folders, manifests and tables; not in git |

| Module | Role |
|---|---|
| `plan.py` | reads the plan and expands experiments into jobs |
| `sites.py` | weather, soil and crop of a site; AquaCrop model of one season |
| `env.py`, `observations.py` | the gymnasium environment shared by DRL agents and rules |
| `drl.py`, `rules.py`, `oracle.py` | PPO training and evaluation, the scheduling rules, the genetic-algorithm oracle |
| `tasks.py` | what one job does and which files it writes |
| `runner.py`, `store.py` | parallel execution, manifests, the result layout |
| `cli.py` | the `ditrl` command |

## Setup

```bash
make install
```

This creates `.venv` with Python 3.10 and every dependency at a pinned version, including the AquaCrop fork at a fixed commit. On Windows the command is then `.venv/Scripts/ditrl`, on Linux `.venv/bin/ditrl`. Training runs on CPU: the crop model is sequential Python, so a GPU does not help.

## Running experiments

```bash
ditrl list stage1                      # the jobs of stage 1 and how many are done
ditrl run smoke smoke_oracle           # every code path in a few minutes
ditrl run stage1                       # stage 1 on all cores but two
ditrl run ablation_a --workers 8       # one experiment
```

A selection is any mix of stages (`stage1`, `stage2`, `stage3`) and experiment names from `config/plan.yaml`. Finished jobs are skipped, so any command can be repeated after an interruption.

**On several machines.** `--shard I/N` runs part I of N equal parts of the job list. The job list depends only on the plan, so every machine derives the same parts. Each machine writes into its own result root (`--results DIR`); the roots merge by copying their `jobs/` folders into one, because no two jobs share a folder. The container image runs the same command:

```bash
make image
docker run --rm -v "$PWD/results:/results" ditrl:<commit> --results /results run stage1 --shard 1/2 --workers 8
```

## Results

```
results/
  jobs/<experiment>/<job id>/   one folder per job
  manifests/                    one file per `ditrl run`: commit, packages, machine, selection, shard
  tables/                       flat tables built by `ditrl collect`
```

A job folder holds `job.json` (the full job: method settings, site, budget, seed), `job.log`, `result.json` (a summary; written last, so its presence marks the job as done) and, depending on the kind of job, `seasons.csv`, `decisions.csv`, `learning.csv`, `model.zip`, `hyperparameters.json`, `grid.csv` or `history.csv` (see `src/ditrl/tasks.py`). Seeds fix every random draw, so a job run again on the same code gives the same numbers.

## From results to the paper

```bash
ditrl collect            # results/tables/: seasons, decisions and learning curves per table of the plan
ditrl assets stage       # figures, tables and macros.tex into paper_assets/
ditrl assets export      # the same, copied into generated/ of the paper repo (path in config/paths.local.yaml)
ditrl assets check       # before submission: do the paper's assets match the current results?
```

`src/ditrl/analysis/assets.py` is the registry of everything that flows into the paper. The oracle reference values come from `data/external/getoptis_ea_*.csv` (published GET-OPTIS values) or from the own oracle, as set in `config/plan.yaml` (`analysis.oracle_source`).
