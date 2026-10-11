# DIT-RL revision experiments

Experiments for the major revision of *Benchmarking deep reinforcement learning strategies for the optimal scheduling of deficit irrigation systems* (Schütze, Kunze, Grové; Agricultural Water Management, AGWAT-D-26-00823): maize irrigation in AquaCrop-OSPy under a hard seasonal water budget, scheduled by two DRL configurations (DRL1 and the reference DRL2 of Kelly et al. 2024), simple rules, and an optimizer with perfect weather foresight.

![System overview](reports/figures/overview.png)

<!-- report:start -->
## Current results

Updated automatically on 2026-10-11 12:34 by `ditrl report`. 1020 of 2372 trainings and rule runs have results. 100% of 1020 ablation runs converged before the step limit; median training length 500k steps. [Progress per experiment](reports/tables/progress.md)

### Central findings



### What changes what

Each panel changes one thing about DRL1. Dot: mean change of test yield against DRL1 trained with the same seed, with 95 % bootstrap interval over seeds; filled when the interval excludes 0. Diamond: the same change averaged over the other choices of the 2 x 4 x 2 design. Cross: planned, results pending.

![What changes what](reports/figures/sensitivity.png)

[Numbers](reports/tables/sensitivity.md) · [Agreed ablation for the co-author](reports/README.md#agreed-ablation) · [All figures and tables](reports/README.md)

<!-- report:end -->

## Quick start

```bash
make install                      # Python 3.10 environment, every dependency pinned
ditrl list stage1                 # jobs of a stage or experiment, and how many are done
ditrl run stage1 --shard 1/2      # run them (resumable); one shard per machine
ditrl export stage1               # result tables -> results/<experiment>/
ditrl report                      # figures and tables -> reports/, and the block above
```

Every DRL run trains until it converges and is tested with its final policy. More on running on several machines, the result layout and the paper assets: [docs/usage.md](docs/usage.md).

## Where things are

| Path | Content |
|---|---|
| `config/plan.yaml` | the whole experiment plan; every number lives here |
| `docs/` | [experiment plan](docs/experiment-plan.md) (why each experiment exists), [decisions](docs/decisions.md), [usage](docs/usage.md) |
| `results/` | result tables per experiment (in git) |
| `reports/` | figures and tables built from them, with an [index](reports/README.md) |
| `src/ditrl/` | the code: plan, environment, training, rules, oracle, runner, export |
| `src/ditrl/report/`, `src/ditrl/analysis/` | the report generators, and the paper's figures, tables and statistics |
| `runs/` | raw job folders with models and logs (not in git) |
