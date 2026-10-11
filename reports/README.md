# Report

Built by `ditrl report` from `results/`. Figures as PNG and PDF, tables as Markdown and CSV.

| File | Shows |
|---|---|
| [figures/overview.png](figures/overview.png) | The two sites, the decision loop of DRL1 with the four design choices the study varies (dials). |
| [tables/progress.md](tables/progress.md) | Every experiment of the plan with its finished runs. |
| [figures/effects_over_budget.png](figures/effects_over_budget.png) | Test yield of the variant minus DRL1 with the same seed, mean over seeds with 95 % bootstrap band (solid), and the same change averaged over the other choices of the 2 x 4 x 2 design (dashed). Gaps: no results for that budget. |
| [tables/sensitivity.md](tables/sensitivity.md) | Change of test yield in t/ha against DRL1 with the 95 % interval, per case. "(k of n)": k of the n planned runs have results. "pending": too few seeds have results yet. Empty: not planned. |
| [tables/core_ablation_configuration.md](tables/core_ablation_configuration.md) | Design choices, PPO settings actually used, training and evaluation of the four variants. |
| [tables/core_ablation_results.md](tables/core_ablation_results.md) | Test yield per case and variant over all seeds, and the paired difference to DRL1 per seed. |
| [figures/core_ablation.png](figures/core_ablation.png) | Mean test yield of DRL1 minus that of each variant, one dot per seed (10 seeds), bar at the mean. Above zero: the change costs yield; below zero: the variant is better than DRL1. |
