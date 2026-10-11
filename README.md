# DIT-RL revision experiments

Experiments for the major revision of *Benchmarking deep reinforcement learning strategies for the optimal scheduling of deficit irrigation systems* (Schütze, Kunze, Grové; Agricultural Water Management, AGWAT-D-26-00823): maize irrigation in AquaCrop-OSPy under a hard seasonal water budget, scheduled by two DRL configurations (DRL1 and the reference DRL2 of Kelly et al. 2024), simple rules, and an optimizer with perfect weather foresight.

![System overview](reports/figures/overview.png)

<!-- report:start -->
## Current results

Updated automatically on 2026-10-11 14:47 by `ditrl report`. 1200 of 2372 trainings and rule runs have results. [Progress per experiment](reports/tables/progress.md)

### Central findings

**1. Rewarding only the yield at the end of the season works better than the stepwise reward of DRL1.** Changing only the reward raises test yield by +1.01 t/ha [+0.55, +1.42] in Nebraska at 75 mm and by +2.39 t/ha [+1.54, +3.27] in Tunis at 300 mm. At every decision the stepwise reward adds the relative evapotranspiration of that day, actual over potential evapotranspiration. This ratio is cheapest to raise early in the season, while the canopy is small, so DRL1 spends its water before flowering and grain filling, the stages in which a water shortage costs the most yield. In Tunis at 150 and 200 mm the crop has died in every run with the stepwise reward so far, in every combination of the other two design choices. Only with a large budget (Tunis 500 mm, +0.08 t/ha [-0.30, +0.57]) do both rewards apply the water at nearly the same time and reach nearly the same yield.

![agreed_irrigation_timing](reports/figures/agreed_irrigation_timing.png)

**2. The 18 observed variables of Kelly et al. (2024) help in Nebraska and hurt in Tunis.** In Nebraska they raise test yield by +1.05 t/ha [+0.65, +1.43] at 75 mm, in Tunis they lower it by -1.47 t/ha [-2.14, -0.56] at 500 mm. Nebraska has rain and weather that differs from year to year, which the extra variables can describe. Tunis is simulated without rain, so there they plausibly add mostly noise. The 18 variables are given to the network without scaling, and the experiment with scaled variables, which has not run yet, will show whether part of the effect comes from the scaling.

**3. How the budget is enforced matters less than the reward, and its effect depends on the case.** Rejecting a dose that exceeds the remaining budget, where DRL1 masks it, changes test yield by -0.15 t/ha [-0.80, +0.50] in Nebraska at 75 mm and by +0.19 t/ha [-0.82, +1.26] in Tunis at 300 mm. Adding a penalty to the rejection helps in Tunis at 300 mm (+1.81 t/ha [+0.95, +2.62]) but costs yield at 600 mm (-1.20 t/ha [-1.95, -0.46, 8 of 10 runs]). Masking, the choice of DRL1, is therefore not better than the alternatives. Where a single change and the same change averaged over the other choices (diamonds in the figure below) differ, the choices interact, most clearly for the reward in Tunis at 500 mm.

### What changes what

Each panel changes one thing about DRL1. Dot: mean change of test yield against DRL1 trained with the same seed, with the 95 % bootstrap interval over seeds, filled when the interval does not contain 0. Diamond: the same change averaged over the other choices of the 2 x 4 x 2 design. Cross: planned, results pending.

![What changes what](reports/figures/sensitivity.png)

[Numbers](reports/tables/sensitivity.md) · [Agreed ablation for the co-author](reports/README.md#agreed-ablation) · [All figures and tables](reports/README.md)

<!-- report:end -->

## Quick start

```bash
make install                      # Python 3.10 environment, every dependency pinned
ditrl list stage1                 # jobs of a stage or experiment, and how many are done
ditrl run stage1 --shard 1/2      # run them, one shard per machine (finished jobs are skipped)
ditrl export stage1               # result tables -> results/<experiment>/
ditrl report                      # figures and tables -> reports/, and the block above
```

Every DRL run is tested with the policy after its last training step. More on running on several machines, the result layout and the paper assets: [docs/usage.md](docs/usage.md).

## Where things are

| Path | Content |
|---|---|
| `config/plan.yaml` | the whole experiment plan, with every number it uses |
| `config/findings.md` | the central findings, written by hand, with their numbers filled in by the report |
| `docs/` | [experiment plan](docs/experiment-plan.md) (why each experiment exists), [decisions](docs/decisions.md), [usage](docs/usage.md) |
| `results/` | result tables per experiment (in git) |
| `reports/` | figures and tables built from them, with an [index](reports/README.md) |
| `src/ditrl/` | the code: plan, environment, training, rules, oracle, runner, export |
| `src/ditrl/report/`, `src/ditrl/analysis/` | the report generators, and the paper's figures, tables and statistics |
| `runs/` | raw job folders with models and logs (not in git) |
