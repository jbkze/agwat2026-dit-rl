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
| `results/` | result tables per experiment, kept in git (below) |
| `runs/` | raw job folders with models and logs; not in git |

| Module | Role |
|---|---|
| `plan.py` | reads the plan and expands experiments into jobs |
| `sites.py` | weather, soil and crop of a site; AquaCrop model of one season |
| `env.py`, `observations.py` | the gymnasium environment shared by DRL agents and rules |
| `drl.py`, `rules.py`, `oracle.py` | PPO training and evaluation, the scheduling rules, the genetic-algorithm oracle |
| `tasks.py` | what one job does and which files it writes |
| `runner.py`, `store.py` | parallel execution, manifests, the layout of the raw job folders |
| `export.py` | the result tables in `results/` |
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
ditrl run ablation_one_change --workers 8
```

A selection is any mix of stages (`stage1`, `stage2`, `stage3`) and experiment names from `config/plan.yaml`. Finished jobs are skipped, so any command can be repeated after an interruption. Every DRL run trains until it converges (`training` in the plan). `--resume-from DIR` continues the trained models of an earlier round found in `DIR/jobs/` instead of starting them anew.

**On several machines.** `--shard I/N` runs part I of N equal parts of the job list. The job list depends only on the plan, so every machine derives the same parts. Each machine writes into its own folder of raw results (`--runs DIR`); these folders merge by copying their `jobs/` and `manifests/` into one, because no two jobs share a folder. The container image runs the same command:

```bash
make image
docker run --rm -v "$PWD/runs:/runs" ditrl:<commit> --runs /runs run stage1 --shard 1/2 --workers 8
```

## Results

Raw results stay outside git, in `runs/` (or wherever `--runs` points):

```
runs/
  jobs/<experiment>/<job id>/   one folder per job: job.json, job.log, result.json, model and tables
  manifests/                    one file per `ditrl run`: commit, packages, machine, selection, shard
```

A job is done when its `result.json` exists; it is written last, together with the commit of the code that produced it. Seeds fix every random draw, so a job run again on the same code gives the same numbers.

`ditrl export <selection>` turns the finished jobs into the tables that are kept in git:

| File in `results/<experiment>/` | Content |
|---|---|
| `jobs.csv` | one row per finished job: method, site, budget, seed, design choices, runtime, code version, summary numbers |
| `seasons.csv` | one row per evaluated season: split, year, yield, water, crop death |
| `learning.csv` | DRL: validation yield over the training steps |
| `decisions.csv.gz` | one row per decision in a test season: day, remaining budget, proposed and applied depth, reward |
| `hyperparameters.csv` | DRL: the PPO settings actually used |

`results/manifests/` holds a copy of every run manifest.

## From results to the paper

```bash
ditrl assets stage       # figures, tables and macros.tex into paper_assets/, from results/
ditrl assets export      # the same, copied into generated/ of the paper repo (path in config/paths.local.yaml)
ditrl assets check       # before submission: do the paper's assets match the current results?
```

`src/ditrl/analysis/assets.py` is the registry of everything that flows into the paper. Which experiments feed which figure or table is set in `config/plan.yaml` (`analysis.tables`). The oracle reference values come from `data/external/getoptis_ea_*.csv` (published GET-OPTIS values) or from the own oracle (`analysis.oracle_source`).
