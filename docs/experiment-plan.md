# Experiment plan for the DIT-RL revision

Status: draft of 2026-09-24. This plan lists every experiment that the reviewers ask for,
directly or through the to-do notes in the response letter (version of 2026-09-22).
Three items are out of scope because no reviewer asks for them: stress tests under a shifted
climate, a second crop model or crop, and extracting a readable rule from a trained policy.

## Shared setup

All experiments use one simulation setup, so that every method sees the same problem.

- **Crop model:** AquaCrop-OSPy from the fork `jbkze/aquacrop-waterstress` (version 3.0.x). The
  fork stores the leaf-expansion stress coefficient on the model state; DRL1 observes it.
- **Tunis site:** sand, soil starts at the permanent wilting point, no rain, maize planted on
  1 May. Training years 1981–1996, validation years 1997–2001, test year 1980.
- **Nebraska site:** silty clay loam, soil starts at 70 % of available water, 100-year synthetic
  weather series from Kelly et al. (2024). Training years 1995–2064, test years 2065–2094. As in
  the paper, the test years also serve for model selection, so that results stay comparable with
  Kelly et al. (2024). This choice is repeated as a limitation in the paper.
- **Seasonal budget:** a hard upper limit. No method may apply more water than the budget.
- **Seeds:** every DRL configuration is trained with 10 seeds. A run is one configuration, one
  budget and one seed.
- **Model selection:** during training the agent is evaluated on the validation years every
  10,000 steps. The checkpoint with the highest mean validation yield is kept and tested.
- **Training length:** 300,000 environment steps for every DRL run, the same for all variants.
  A first check with 100,000 steps stopped DRL1 while its validation yield was still rising
  (`docs/decisions.md`). The learning curves from experiment E6 show whether 300,000 is enough.
- **Hyperparameters:** DRL1 uses the PPO defaults of stable-baselines3 2.3.2 and sb3-contrib
  2.3.0, as in the thesis. Kelly et al. trained DRL2 with Ray RLlib 1.13; DRL2 here uses the
  stable-baselines3 equivalents of that configuration (discount factor 1, 3 hidden layers of
  64 units, 512 steps per update, the RLlib defaults for the rest). All DRL variants therefore
  run in the same library, so that library differences do not enter the comparisons. The exact
  values are written to every run folder and go into the parameter table of the paper (R2#3).

The configurations DRL1 and DRL2 are defined in the unified environment as follows.

| Setting | DRL1 | DRL2 (Kelly et al. 2024) |
|---|---|---|
| Observation | 5 variables: remaining budget, root-zone water, deep percolation, leaf-expansion stress, growth stage | Tunis: same 5 variables (as in the paper). Nebraska: the 18 variables of Kelly et al. |
| Action | discrete depths 0, 1, 2, 3, 5, 8, 13, 21, 34 mm | continuous depth 0–34 mm (Tunis) or 0–25 mm (Nebraska) |
| Decision interval | 5 days | 7 days (Tunis) or 1 day (Nebraska) |
| Budget handling | action mask: depths above the remaining budget cannot be chosen | depth is cut to the remaining budget, no penalty |
| Reward | relative evapotranspiration at each step, yield at the end | yield (Tunis) or profit (Nebraska) at the end only |
| Algorithm | MaskablePPO (sb3-contrib), default settings | PPO (stable-baselines3), settings of Kelly et al. |

The DRL2 settings above follow the manuscript text and the code of Kelly et al. Before the
campaign starts they are checked once more against the archived reference code (open point).

## Experiments

### E0: Perfect-foresight oracle and robust open-loop schedule

Two optimizer runs that need no learning:

- **EA-DS:** for every test year and budget, the schedule that maximizes yield when the weather
  of that year is known in advance. Tunis: year 1980, 13 budgets from 0 to 800 mm. Nebraska:
  30 test years, budgets 75, 100 and 125 mm.
- **EA-SS:** one fixed schedule for all years that maximizes the 10 % quantile of yield over the
  years 1981–2001 (training and validation years, as in the paper). Tunis only, at the five
  budgets of E2. For Nebraska, EA-SS is optional because of
  its cost (see compute estimate).

The optimizer is a genetic algorithm from pymoo with an event-based encoding: a fixed number of
irrigation events, each with a day and a depth. Depths are scaled down when the sum exceeds the
budget. To make sure the search converges:

- a run stops only when the best yield has improved by less than 0.001 t/ha over the last
  50 generations; a cap of 1,000 generations exists only as a safeguard and is logged when hit,
- every case is solved with several restarts (number of events 6, 10 and 20, three seeds each),
  and the best restart is kept,
- a case counts as converged when the two best restarts differ by less than 0.05 t/ha;
  otherwise it is flagged and solved again with more restarts,
- the best yield per generation is logged, so that convergence curves can go into the appendix.

Niels will provide the GET-OPTIS code. Once GET-OPTIS results exist, they are stored as a CSV
file in `data/external/` and the analysis reads them through one entry in the config
(`oracle.source`). All other experiments stay unchanged.

Answers: R1#2 (role of the oracle), R2#4 (together with E3).

### E1: Benchmark across budgets

All methods at all budgets: DRL1, DRL2, two simple heuristics under the same budget, full
irrigation without a budget, EA-SS and EA-DS.

- **Threshold heuristic:** at each decision the root zone is refilled when its depletion exceeds
  a fraction of the total available water; the depth is cut to the remaining budget. The
  fraction is chosen by grid search on the training years, separately for each budget.
- **Fixed-interval heuristic:** the budget is split into equal doses at evenly spaced dates. The
  number of doses is chosen by grid search on the training years.
- **Simplex heuristics by Niels:** imported from `data/external/` as they are.

Output: crop water production function per site with the spread over seeds, yield as a share
of EA-DS, water actually used, and whether yield rises with the budget.

Answers: R1#5 (simple heuristics), R3#3 (together with E4), RQ (i).

### E2: Ablation of the three design choices

Full factorial design over the three components of DRL1. Action set and decision interval are
held at the DRL1 values, so that masking is possible in every cell.

| Factor | Level 1 (DRL1) | Level 2 |
|---|---|---|
| Observation | 5 variables | 18 variables of Kelly et al. |
| Budget handling | action mask | cut to remaining budget |
| Reward | stepwise relative evapotranspiration + yield | yield at the end only |

Budgets: Tunis 150, 200, 300, 400, 600 mm; Nebraska 75, 100, 125 mm. Eight cells, 10 seeds
each. The cell "5 variables, mask, stepwise reward" has the same settings as DRL1 in E1; it is
trained again inside E2, so that the ablation is one self-contained run and the repetition shows
how well DRL1 results reproduce.

Analysis: the main effect of each factor on test yield, paired over the other factors, seeds and
test years, with a bootstrap interval, a Wilcoxon signed-rank test and Holm correction over all
factors, sites and budgets (`analysis/stats.py`).

Answers: R1#3, R2#2, R3#4.

### E3: Value of weather information

Splits the gap between DRL1 and EA-DS into a part caused by missing weather information and a
part caused by the learning method. Two extra DRL variants:

- **DRL1 with forecast:** DRL1 plus the mean reference evapotranspiration and mean rainfall of
  the next 7 days, taken from the weather file (a perfect 7-day forecast, as in the `forecast`
  observation set of Kelly et al.). Budgets as in E2, 10 seeds.
- **DRL1 trained on the test year:** trained and tested on the same single year, as in the
  year-specific reference of Saikai et al. (2023). Tunis: 1980, budgets as in E2, 10 seeds.
  Nebraska: all 30 test years, three budgets, 3 seeds per year.

Result: an ordered comparison per budget, from no weather knowledge to full knowledge: EA-SS,
heuristics, DRL1, DRL1 with forecast, DRL1 trained on the test year, EA-DS. The difference
between neighbouring rows is reported as the value of that piece of information; it holds for
this model setup only.

Answers: R2#4 (decomposition of the gap).

### E4: Why DRL fails at low budgets in Tunis

At 100–200 mm both DRL agents produce zero yield, while EA-DS finds productive schedules. The
oracle schedules from E0 apply a few large doses. The hypothesis: the DRL1 action set (at most
34 mm every 5 days) cannot reproduce such doses early enough. Two factors on top of DRL1:

| Factor | Levels |
|---|---|
| Action set | 0–34 mm (DRL1); 0, 5, 13, 21, 34, 55, 89 mm |
| Decision interval | 5 days (DRL1); 1 day |

Budgets 100, 125, 150 and 200 mm, 10 seeds. In addition, every test episode of E1 and E4 logs
the proposed action, the applied depth and the remaining budget at each decision, so that the
distribution of chosen actions can be shown for 100–200 mm.

Answers: R3#3.

### E5: Robustness and extreme years

No new training. The trained models from E1 and E2 are evaluated on:

- **each test year separately:** per-year yields, the 10 % quantile, and the mean of the worst
  10 % of years (CVaR); the three driest Nebraska test years (lowest seasonal rainfall minus
  reference evapotranspiration) are shown individually,
- **noisy observations:** Gaussian noise with a standard deviation of 5 %, 10 % and 20 % of the
  value on root-zone water and deep percolation, 10 noise seeds per test year.

The analysis keeps apart zero yield, crop death and a missing result.

Answers: R1#4, R2#5 (behaviour under observation errors).

### E6: Learning curves and reliability over seeds

No new training. From the validation log of every run of E1 to E4: mean validation yield over
training steps, and the share of seeds that reach at least 90 % of the best seed.

Answers: R2#3 (training details), supports R1#3.

### E7: Ex post economics

No new training. Profit per test year from yield and water use, with the prices of Kelly et al.
(2024): 180 USD per tonne of maize, 1 USD per mm and hectare of irrigation, 1,728 USD per
hectare of fixed costs. Prices are an evaluation setting only; no method optimizes profit
except DRL2 in Nebraska, which uses it as its reward (as in Kelly et al.).

## Compute estimate

Measured on the h200 server in July 2026: one DRL1 training with 100,000 steps takes about
7 minutes on one CPU core (about 10 minutes on a laptop core in September 2026), so about
21 minutes with 300,000 steps; the GPU does not speed it up. About 60 runs fit in parallel. The
oracle evaluates about 77 schedules per second on 56 cores.

| Experiment | Runs or evaluations | Time on h200 |
|---|---|---|
| E1 DRL | 320 trainings | about 2 h |
| E2 | 640 trainings | about 4 h |
| E3 | 400 trainings | about 2.5 h |
| E4 | 160 trainings | about 1 h |
| E5, E6, E7 | evaluation only | minutes |
| E0 EA-DS Tunis + Nebraska | about 6.6 million simulations | about 1 day |
| E0 EA-SS Tunis (5 budgets, 21 years each) | about 6.8 million simulations | about 1 day |
| E0 EA-SS Nebraska (optional) | about 13.6 million simulations | about 2 days |

The oracle dominates the compute. With GET-OPTIS results from Niels, E0 is not needed.

## Mapping to the reviewer comments

| Comment | Experiment |
|---|---|
| R1#2 oracle role | E0, E3 |
| R1#3 ablation | E2 |
| R1#4 robustness, extreme cases | E5 |
| R1#5 simple heuristics | E1 |
| R2#2 benefit of each component | E2 |
| R2#3 hyperparameters | run manifests, E6 |
| R2#4 decomposition of the gap | E3 |
| R2#5 path to the field | E5 (observation noise) |
| R3#3 actions at 100–200 mm | E4, action logs of E1 |
| R3#4 ablation | E2 |
| R3#6 masking pseudocode | `src/ditrl/env.py`, method `action_masks` |
