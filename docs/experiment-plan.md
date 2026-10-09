# Experiment plan for the DIT-RL revision

Status: 2026-10-09 (ablation rebuilt around the variants agreed with the co-author, growing to the full design, and every DRL method now evaluates its last policy). Every number of this plan (sites, budgets, seeds, methods, settings) is defined in `config/plan.yaml`; this document says why each experiment exists. The plan is built in three stages. Stage 1 contains the minimum the revision needs; stages 2 and 3 add to it without repeating anything. Each stage ends with a complete set of results, so the campaign can stop after any stage, depending on the time left.

It covers every experiment that the reviewers ask for, directly or through the to-do notes in the response letter (version of 2026-09-22). Three items are out of scope because no reviewer asks for them: stress tests under a shifted climate, a second crop model or crop, and extracting a readable rule from a trained policy.

## Shared setup

All experiments use one simulation setup, so that every method sees the same problem.

- **Crop model:** AquaCrop-OSPy from the fork `jbkze/aquacrop-waterstress` (3.0.8, pinned to commit 818b781). The fork stores the leaf-expansion stress coefficient on the model state; DRL1 observes it.
- **Tunis site:** sand, soil starts at the permanent wilting point, no rain, maize planted on 1 May. Training years 1981–1996, validation years 1997–2001, test year 1980.
- **Nebraska site:** silty clay loam, soil starts at 70 % of available water, 100-year synthetic weather series from Kelly et al. (2024). Training years 1995–2064, test years 2065–2094. The paper and Kelly et al. (2024) also selected the model on the test years. This plan does not (see model selection below).
- **Seasonal budget:** a hard upper limit. No method may apply more water than the budget.
- **Seeds:** 5 per DRL configuration in stage 1, 10 from stage 2 on (stage 2 adds seeds 5-9). A training job is one configuration, one budget and one seed.
- **Model selection:** none. Every DRL run is tested with the policy after its last training step, with deterministic actions, and no checkpoint or hyperparameter is chosen on evaluation data (decided on 2026-10-09). The agent is still evaluated on the validation years every 10,000 steps, but only for the learning curves (E6). Runs before that date kept the checkpoint with the best validation yield.
- **Training length:** 300,000 environment steps for every DRL run, the same for all variants. A first check with 100,000 steps stopped DRL1 while its validation yield was still rising (`docs/decisions.md`). The learning curves from experiment E6 show whether 300,000 is enough.
- **Hyperparameters:** DRL1 uses the PPO defaults of stable-baselines3 2.3.2 and sb3-contrib 2.3.0, as in the thesis. Kelly et al. trained DRL2 with Ray RLlib 1.13; DRL2 here uses the stable-baselines3 equivalents of that configuration (discount factor 1, 3 hidden layers of 64 units, 512 steps per update, the RLlib defaults for the rest). All DRL variants therefore run in the same library, so that library differences do not enter the comparisons. The exact values are written to every run folder and go into the parameter table of the paper (R2#3).

The configurations DRL1 and DRL2 are defined in the unified environment as follows.

| Setting | DRL1 | DRL2 (Kelly et al. 2024) |
|---|---|---|
| Observation | 5 variables: remaining budget, root-zone water, deep percolation, leaf-expansion stress, growth stage | Tunis: same 5 variables (as in the paper). Nebraska: the 18 variables of Kelly et al. |
| Action | discrete depths 0, 1, 2, 3, 5, 8, 13, 21, 34 mm | continuous depth 0–34 mm (Tunis) or 0–25 mm (Nebraska) |
| Decision interval | 5 days | 5 days (Tunis, as in the code) or 1 day (Nebraska) |
| Budget handling | action mask: depths above the remaining budget cannot be chosen | depth is cut to the remaining budget, no penalty |
| Reward | relative evapotranspiration at each step, yield at the end | yield (Tunis) or profit (Nebraska) at the end only; see `docs/diagnosis-drl2.md` for the unscaled profit reward of the published Tunis runs |
| Algorithm | MaskablePPO (sb3-contrib), default settings | PPO (stable-baselines3), settings of Kelly et al. |

The DRL2 settings above follow the manuscript text and the Tunis code that produced the published values. That code rewarded DRL2 with the unscaled profit, which explains its weak published results in Tunis (`docs/diagnosis-drl2.md`); DRL2 here uses the yield reward the manuscript describes.

## Stages at a glance

| Stage | What it adds | DRL trainings | Laptop (14 workers) | Two servers, 8 cores each |
|---|---|---|---|---|
| 1 Minimum | benchmark at 8 budgets, rules, 5 seeds; the agreed ablation (DRL1 and three variants, three cases) with its pilot | 140 (+ 12 in the pilot) | about 17 h | about 6 h |
| 2 Standard | 10 seeds, all Tunis budgets, low-budget diagnosis, noise; the rest of the ablation up to the full 2 x 4 x 2 design | 1,720 | about 8 days | about 2.5 days |
| 3 Complete | value of weather information, scaling check, own pymoo oracle | 480 + 972 oracle restarts | about 5 days | about 15 h |

Training times are measured: on the laptop, 14 parallel trainings of 300,000 steps take about 100 minutes, i.e. about 8 trainings per hour; measured in the pilot of the ablation on 2026-10-09 with all runs of a job side by side on a server with 8 cores per job: 30 to 35 minutes per Tunis training (8 at once) and 24 to 27 minutes per Nebraska training (4 at once), so two such servers manage about 27 trainings per hour. The code was about 25 % faster per training after that measurement (season starts copied instead of rebuilt), which the table does not count yet. The oracle times in stage 3 are a rough estimate from about 0.1 to 0.15 seconds per simulated season. Rules and evaluations take minutes and are left out of the table.

Run a stage with `ditrl run stage1`, or part of it on several machines with `ditrl run stage1 --shard 1/2` and `--shard 2/2`. Finished jobs are skipped, so every call can be repeated after an interruption; `ditrl list stage1` shows what is done.

**Oracle values in stages 1 and 2.** The EA-DS and EA-SS values of the manuscript are GET-OPTIS results (Tunis: Tab. B1; Nebraska: EA-DS per test year). They are stored in `data/external/` and serve as the reference until the GET-OPTIS code is available. Stage 3 adds an own oracle with pymoo, which can check or replace them.

## Stage 1: Minimum

Enough to answer the requests that the reviewers share: ablation, simple rules, parameter documentation, and a DRL2 that is trained correctly.

| Experiment | Content | Jobs |
|---|---|---|
| `benchmark` | DRL1, DRL2, threshold rule, fixed-interval rule, full irrigation; Tunis 150, 200, 300, 400, 600 mm; Nebraska 75, 100, 125 mm; 5 seeds | 80 DRL + 18 rules |
| `ablation_pilot` | pilot of the agreed ablation: every variant once per case, seed 100, to check the code and measure the runtime (done on 2026-10-09; not part of a stage) | 12 DRL |
| `ablation_one_change` | the agreed ablation (E2): DRL1 and three variants that change one design choice each, Tunis 300 and 500 mm, Nebraska 75 mm, seeds 0-4 | 60 DRL |

Without extra runs, stage 1 also gives: the hyperparameter table (R2#3), the masking pseudocode (R3#6), the actions chosen at 150 and 200 mm (first part of R3#3), per-year yields and the lower tail in Nebraska (first part of R1#4), learning curves (E6) and the ex post profit (E7). The diagnosis runs for DRL2 are done (`docs/diagnosis-drl2.md`).

Answers: R1#3, R2#2, R3#4 (one-factor ablation), R1#5, R2#3, R3#6; partly R1#4 and R3#3.

## Stage 2: Standard

Makes stage 1 statistically solid and complete across budgets.

| Experiment | Content | Jobs |
|---|---|---|
| `benchmark_seeds` | seeds 5-9 for DRL1 and DRL2 of stage 1 | 80 DRL |
| `ablation_budget_handling` | ablation (E2): cut and penalty as further budget handling, and the full Kelly cell | 45 DRL |
| `ablation_interactions` | ablation (E2): the nine cells that complete the 2 x 4 x 2 design | 135 DRL |
| `benchmark_budgets` | the remaining Tunis budgets 100, 125, 500, 700, 725, 750, 800 mm (0 mm gives 0 t/ha for every method and needs no run); 10 seeds | 140 DRL + 14 rules |
| `low_budget` | E4 below, 10 seeds | 120 DRL |
| `robustness` | benchmark models of stages 1 and 2 with noisy observations | evaluation only |
| `ablation_more_seeds`, `ablation_more_budgets` | ablation (E2): seeds 5-9 for all 16 cells, then all 16 cells at the other budgets of the ablation | 240 + 960 DRL |

Answers in addition: R1#4 and R2#5 (noise), R3#3 (why DRL fails at low budgets), and the full crop water production function of RQ (i).

## Stage 3: Complete

Adds what makes the paper stronger but is not strictly needed for the revision.

| Experiment | Content | Jobs |
|---|---|---|
| `forecast` | DRL1 with a perfect 7-day forecast (E3) | 80 DRL |
| `test_year` | DRL1 trained on the test year itself (E3) | 320 DRL |
| `obs_scaled` | 18 variables with fixed scaling: is the observation effect of the ablation partly a scaling effect? | 80 DRL |
| `oracle_ds`, `oracle_ss` | own pymoo oracle (E0), needed only without GET-OPTIS code or as a check | 927 + 45 restarts |

Answers in addition: R2#4 (decomposition of the gap), R1#2 (role of the oracle, with our own convergence check).

## Experiments in detail

### E0: Perfect-foresight oracle and robust open-loop schedule (stage 3)

Two optimizer runs that need no learning:

- **EA-DS:** for every test year and budget, the schedule that maximizes yield when the weather of that year is known in advance. Tunis: year 1980, 13 budgets from 0 to 800 mm. Nebraska: 30 test years, budgets 75, 100 and 125 mm.
- **EA-SS:** one fixed schedule for all years that maximizes the 10 % quantile of yield over the years 1981–2001 (training and validation years, as in the paper). Tunis only, at the five budgets of E2. For Nebraska, EA-SS is optional because of its cost (see compute estimate).

The optimizer is a genetic algorithm from pymoo with an event-based encoding: a fixed number of irrigation events, each with a day and a depth. Depths are scaled down when the sum exceeds the budget. To make sure the search converges:

- a run stops only when the best yield has improved by less than 0.001 t/ha over the last 50 generations; a cap of 1,000 generations exists only as a safeguard and is logged when hit,
- every case is solved with several restarts (number of events 6, 10 and 20, three seeds each), and the best restart is kept,
- a case counts as converged when the two best restarts differ by less than 0.05 t/ha; otherwise it is flagged and solved again with more restarts,
- the best yield per generation is logged, so that convergence curves can go into the appendix.

The GET-OPTIS code arrived from the co-author on 2026-10-09. New GET-OPTIS results are stored as CSV files in `data/external/`; the analysis reads the reference values from the source named in `config/plan.yaml` (`analysis.oracle_source`: the GET-OPTIS files, or the own oracle). All other experiments stay unchanged.

Answers: R1#2 (role of the oracle), R2#4 (together with E3).

### E1: Benchmark across budgets (stage 1, completed in stage 2)

All methods at all budgets: DRL1, DRL2, two simple heuristics under the same budget, full irrigation without a budget, EA-SS and EA-DS.

- **Threshold heuristic:** at each decision the root zone is refilled when its depletion exceeds a fraction of the total available water; the depth is cut to the remaining budget. The fraction is chosen by grid search on the training years, separately for each budget.
- **Fixed-interval heuristic:** the budget is split into equal doses at evenly spaced dates. The number of doses is chosen by grid search on the training years.
- **Simplex heuristics by Niels:** imported from `data/external/` as they are.

Output: crop water production function per site with the spread over seeds, yield as a share of EA-DS, water actually used, and whether yield rises with the budget.

Answers: R1#5 (simple heuristics), R3#3 (together with E4), RQ (i).

### E2: Ablation of the three design choices (agreed part in stage 1, the rest in stage 2)

The ablation grows in four parts. The first is the part agreed with the co-author and is required. Every later part adds cells, seeds or budgets without repeating anything, so the ablation can stop after any step. Action set and decision interval stay at the DRL1 values throughout, so that masking is possible in every cell.

| Factor | Levels |
|---|---|
| Observation | 5 variables (DRL1); 18 variables of Kelly et al. |
| Budget handling | action mask (DRL1); reject: a depth above the remaining budget is not applied (0 mm), no penalty; cut to the remaining budget, no penalty (Kelly et al.); penalty: as reject, and 1 is subtracted from the reward of that step |
| Reward | relative evapotranspiration at each step + yield at the end (DRL1); yield at the end only |

The 18 variables do not contain the 5: only the remaining budget and the growth stage (one-hot at Kelly) are in both, while root-zone water, deep percolation and expansion stress are missing at Kelly. Changing the observation therefore compares two kinds of information, not two numbers of variables, and the paper says so.

With reject and penalty the policy proposes a depth u_t and the simulator applies a_t = u_t if u_t fits into the remaining budget B_t, else 0. The PPO update uses the proposed depths, the water accounting the applied ones, and the decision log keeps both.

The mask follows Huang and Ontañón (2022, FLAIRS-35): masked depths get probability 0 when an action is sampled, and the masks are stored with each transition and applied again when the PPO update computes log-probabilities, ratios and entropy (MaskablePPO, sb3-contrib 2.3.0). At evaluation the agent takes the most probable feasible depth. The remaining budget is also part of the observation. If a masked depth ever reached the simulator, the environment stops with an error instead of cutting it, so DRL1 cannot turn into the cut variant unnoticed. In the terms of Krasowski et al. (2023, TMLR) the four budget handlings are masking, projection (cut), replacement by 0 mm (reject), and replacement with a penalty.

| Part | Experiment | Cells | Cases | Seeds | Runs |
|---|---|---|---|---|---|
| Pilot | `ablation_pilot` | the four of the agreed part | as in the agreed part | 100 | 12 |
| Agreed, required | `ablation_one_change` | DRL1; 18 variables; reject; reward at the end only | Tunis 300 mm, Tunis 500 mm (test season 1980), Nebraska 75 mm (test years 71-100) | 0-4 | 60 |
| Budget handling | `ablation_budget_handling` | cut; penalty; the full Kelly cell (18 variables, cut, reward at the end) | as in the agreed part | 0-4 | 45 |
| Interactions | `ablation_interactions` | the nine cells that complete the 2 x 4 x 2 design | as in the agreed part | 0-4 | 135 |
| More seeds and budgets | `ablation_more_seeds`, `ablation_more_budgets` | all 16 cells | as in the agreed part, then Tunis 150, 200, 400, 600 mm and Nebraska 100, 125 mm | 5-9, then 0-9 | 1,200 |

Before the agreed part runs: the training budget, PPO hyperparameters, network, reward scaling, discounting, normalization and software versions are written into the configuration table from the run folders (`hyperparameters.json`, `manifest.json`), and the pilot gives the runtime. A run that fails is restarted with the same seed and reported. A seed is never replaced by another one.

Evaluation per run: mean test yield (1 season in Tunis, 30 in Nebraska) of the final policy with deterministic actions. Per variant and case, the paired difference to DRL1 over the same seed; positive values mean that the change costs yield. Nebraska in addition: the 10 % quantile of yield per trained policy (for 30 years the fourth-lowest value, as in the main analysis), then summarized across seeds. Years and seeds are not pooled into one distribution. Once the interactions have run, the main effect of each factor is paired over the other factors, with a bootstrap interval, a Wilcoxon signed-rank test and Holm correction over all factors, sites and budgets (`src/ditrl/analysis/stats.py`).

Reporting: configuration table, result table over all cells and cases, and a plot of the paired differences per comparison. The spread across seeds is always shown.

An earlier 2 x 2 x 2 design with the best checkpoint (configs up to commit e5de53f) is superseded.

Answers: R1#3, R2#2, R3#4.

### E3: Value of weather information (stage 3)

Splits the gap between DRL1 and EA-DS into a part caused by missing weather information and a part caused by the learning method. Two extra DRL variants:

- **DRL1 with forecast:** DRL1 plus the mean reference evapotranspiration and mean rainfall of the next 7 days, taken from the weather file (a perfect 7-day forecast, as in the `forecast` observation set of Kelly et al.). Budgets as in E2, 10 seeds.
- **DRL1 trained on the test year:** trained and tested on the same single year, as in the year-specific reference of Saikai et al. (2023). Tunis: 1980, budgets as in E2, 10 seeds. Nebraska: all 30 test years, three budgets, 3 seeds per year.

Result: an ordered comparison per budget, from no weather knowledge to full knowledge: EA-SS, heuristics, DRL1, DRL1 with forecast, DRL1 trained on the test year, EA-DS. The difference between neighbouring rows is reported as the value of that piece of information; it holds for this model setup only.

Answers: R2#4 (decomposition of the gap).

### E4: Why DRL fails at low budgets in Tunis (stage 2)

At 100–200 mm both DRL agents produce zero yield, while EA-DS finds productive schedules. The oracle schedules from E0 apply a few large doses. The hypothesis: the DRL1 action set (at most 34 mm every 5 days) cannot reproduce such doses early enough. Two factors on top of DRL1:

| Factor | Levels |
|---|---|
| Action set | 0–34 mm (DRL1); 0, 5, 13, 21, 34, 55, 89 mm |
| Decision interval | 5 days (DRL1); 1 day |

Budgets 100, 125, 150 and 200 mm, 10 seeds. In addition, every test episode of E1 and E4 logs the proposed action, the applied depth and the remaining budget at each decision, so that the distribution of chosen actions can be shown for 100–200 mm.

Answers: R3#3.

### E5: Robustness and extreme years (per-year results from stage 1, noise in stage 2)

No new training. The trained benchmark models are evaluated on:

- **each test year separately:** per-year yields, the 10 % quantile, and the mean of the worst 10 % of years (CVaR); the three driest Nebraska test years (lowest seasonal rainfall minus reference evapotranspiration) are shown individually,
- **noisy observations:** Gaussian noise with a standard deviation of 5 %, 10 % and 20 % of the value on root-zone water and deep percolation, 10 noise seeds per test year.

The analysis keeps apart zero yield, crop death and a missing result.

Answers: R1#4, R2#5 (behaviour under observation errors).

### E6: Learning curves and reliability over seeds (from stage 1)

No new training. From the validation log of every run of E1 to E4: mean validation yield over training steps, and the share of seeds that reach at least 90 % of the best seed.

Answers: R2#3 (training details), supports R1#3.

### E7: Ex post economics (from stage 1)

No new training. Profit per test year from yield and water use, with the prices of Kelly et al. (2024): 180 USD per tonne of maize, 1 USD per mm and hectare of irrigation, 1,728 USD per hectare of fixed costs. Prices are an evaluation setting only; no method optimizes profit except DRL2 in Nebraska, which uses it as its reward (as in Kelly et al.).

## Mapping to the reviewer comments

| Comment | Experiment | Answered from stage |
|---|---|---|
| R1#2 oracle role | E0, E3 | wording in any stage; own check in 3 |
| R1#3 ablation | E2 | 1 (main effects), 2 (interactions) |
| R1#4 robustness, extreme cases | E5 | 1 (per-year tails), 2 (noise) |
| R1#5 simple heuristics | E1 | 1 |
| R2#2 benefit of each component | E2 | 1 |
| R2#3 hyperparameters | run manifests, E6 | 1 |
| R2#4 decomposition of the gap | E3 | 3 |
| R2#5 path to the field | E5 (observation noise) | 2 |
| R3#3 actions at 100-200 mm | E4, action logs of E1 | 1 (actions), 2 (cause) |
| R3#4 ablation | E2 | 1 |
| R3#6 masking pseudocode | `src/ditrl/env.py`, method `action_masks` | 1 |
