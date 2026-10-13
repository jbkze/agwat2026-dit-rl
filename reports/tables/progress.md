| Stage | Experiment | What it tests | Runs | Results |
|---|---|---|---|---|
| 1 | `benchmark` | DRL1, DRL2 and the rules over the budgets, 5 seeds | 98 of 98 | [results/benchmark/](../../results/benchmark/) |
| 1 | `ablation_one_change` | core part: DRL1 and three variants that each change one design choice | 60 of 60 | [results/ablation_one_change/](../../results/ablation_one_change/) |
| 2 | `ablation_budget_handling` | the other two budget limits, and all three settings of Kelly et al. at once | 45 of 45 | [results/ablation_budget_handling/](../../results/ablation_budget_handling/) |
| 2 | `ablation_interactions` | the remaining cells of the 2 x 4 x 2 design | 135 of 135 | [results/ablation_interactions/](../../results/ablation_interactions/) |
| 2 | `benchmark_seeds` | DRL1 and DRL2 with 5 more seeds | 80 of 80 | [results/benchmark_seeds/](../../results/benchmark_seeds/) |
| 2 | `benchmark_budgets` | the rest of the yield over budget curve in Tunis | 154 of 154 | [results/benchmark_budgets/](../../results/benchmark_budgets/) |
| 2 | `low_budget` | larger doses and daily decisions at small budgets | 120 of 120 | [results/low_budget/](../../results/low_budget/) |
| 2 | `robustness` | the benchmark policies with noisy observations | 4800 of 4800 | [results/robustness/](../../results/robustness/) |
| 2 | `ablation_more_seeds` | the 16 cells of the design with 5 more seeds | 240 of 240 | [results/ablation_more_seeds/](../../results/ablation_more_seeds/) |
| 2 | `ablation_more_budgets` | the 16 cells of the design at the other budgets | 960 of 960 | [results/ablation_more_budgets/](../../results/ablation_more_budgets/) |
| 3 | `forecast` | DRL1 with a perfect 7-day weather forecast | 80 of 80 | [results/forecast/](../../results/forecast/) |
| 3 | `test_year` | DRL1 trained on the test year itself | 320 of 320 | [results/test_year/](../../results/test_year/) |
| 3 | `obs_scaled` | the 18 variables, scaled | 8 of 80 | [results/obs_scaled/](../../results/obs_scaled/) |
| 3 | `oracle_ds` | best schedule per test year with known weather (not run, published values used) | 0 of 927 | pending |
| 3 | `oracle_ss` | one robust schedule for all years (not run, published values used) | 0 of 45 | pending |
