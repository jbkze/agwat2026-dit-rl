# Usage

Details behind the quick start of the [README](../README.md).

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

`ditrl export <selection>` turns the finished jobs into the tables that are kept in git, and `ditrl report` rebuilds the figures and tables in `reports/` and the results block of the README from them:

| File in `results/<experiment>/` | Content |
|---|---|
| `jobs.csv` | one row per finished job: method, site, budget, seed, design choices, runtime, code version, summary numbers |
| `seasons.csv` | one row per evaluated season: split, year, yield, water, crop death |
| `learning.csv` | DRL: validation yield over the training steps |
| `decisions.csv.gz` | one row per decision in a test season: day, remaining budget, proposed and applied depth, reward |
| `hyperparameters.csv` | DRL: the PPO settings actually used |
| `convergence.csv` | DRL: mean yield of the training seasons per block of steps, the input of the stopping rule |
| `training.csv.gz` | DRL: PPO statistics (episode return, entropy, losses, KL, explained variance) about every 10,000 steps |

`results/manifests/` holds a copy of every run manifest.

## From results to the paper

```bash
ditrl assets stage       # figures, tables and macros.tex into paper_assets/, from results/
ditrl assets export      # the same, copied into generated/ of the paper repo (path in config/paths.local.yaml)
ditrl assets check       # before submission: do the paper's assets match the current results?
```

`src/ditrl/analysis/assets.py` is the registry of everything that flows into the paper. Which experiments feed which figure or table is set in `config/plan.yaml` (`analysis.tables`). The oracle reference values come from `data/external/getoptis_ea_*.csv` (published GET-OPTIS values) or from the own oracle (`analysis.oracle_source`).
