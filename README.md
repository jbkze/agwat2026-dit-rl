# DIT-RL revision experiments

Experiments for the major revision of *Benchmarking deep reinforcement learning strategies for the optimal scheduling of deficit irrigation systems* (Schütze, Kunze, Grové; Agricultural Water Management, AGWAT-D-26-00823): maize irrigation in AquaCrop-OSPy under a hard seasonal water budget, scheduled by two DRL configurations (DRL1 and the reference DRL2 of Kelly et al. 2024), simple rules, and an optimizer with perfect weather foresight.

![System overview](reports/figures/overview.png)

<!-- report:start -->
## Current results

Updated automatically on 2026-10-10 23:50 by `ditrl report`; all figures and tables are in [reports/](reports/README.md).

- 780 of 2372 trainings and rule runs have results.
- Nebraska 75 mm: best variant 18 variables with 12.00 t/ha against 10.95 t/ha for DRL1 (10 seeds).
- Tunis 300 mm: best variant yield at the end with 4.22 t/ha against 1.82 t/ha for DRL1 (10 seeds).
- Tunis 500 mm: best variant yield at the end with 9.78 t/ha against 9.70 t/ha for DRL1 (10 seeds).
- Ablation so far: 780 of 1520 runs; effects at budgets with few runs are preliminary.
- Switching to 18 variables changes test yield significantly in: Nebraska 75 mm +0.64, Nebraska 100 mm +0.61, Nebraska 125 mm +0.67, Tunis 300 mm +0.46 t/ha.
- Switching to cut to remaining budget changes test yield significantly in: Nebraska 75 mm -0.15, Nebraska 100 mm +1.99 t/ha.
- Switching to reject + penalty changes test yield significantly in: Nebraska 75 mm -0.24, Nebraska 100 mm +0.41 t/ha.
- Switching to yield at the end changes test yield significantly in: Nebraska 75 mm +0.45, Nebraska 100 mm +0.61, Nebraska 125 mm +0.74, Tunis 150 mm +1.12, Tunis 200 mm +2.06, Tunis 300 mm +1.84, Tunis 400 mm +1.85, Tunis 500 mm +0.81 t/ha.
- 100% of 780 ablation runs converged before the step limit; median training length 501k steps.

[Findings so far](reports/findings.md) · [Progress per experiment](reports/tables/progress.md)

**Agreed ablation, paired differences (10 seeds).** Mean test yield of DRL1 minus that of each variant, one dot per seed (10 seeds), bar at the mean. Above zero: the change costs yield; below zero: the variant is better than DRL1.

![Agreed ablation, paired differences (10 seeds)](reports/figures/agreed_ablation_10_seeds.png)

**Agreed ablation, paired differences (5 seeds).** Mean test yield of DRL1 minus that of each variant, one dot per seed (5 seeds), bar at the mean. Above zero: the change costs yield; below zero: the variant is better than DRL1.

![Agreed ablation, paired differences (5 seeds)](reports/figures/agreed_ablation_5_seeds.png)

**Ablation, main effects.** Change of mean test yield when one design choice of DRL1 is switched, paired over the other choices, seeds and test years, with 95 % bootstrap intervals; one line per site over the seasonal budget.

![Ablation, main effects](reports/figures/ablation_main_effects.png)

**Agreed ablation, learning curves.** Mean yield on the validation years during training, median and interquartile range over seeds, averaged in bins of 50,000 steps, drawn while at least half of the runs are still training. In Nebraska the validation years are the test years; nothing is selected on them.

![Agreed ablation, learning curves](reports/figures/agreed_learning_curves.png)

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
