# Why the published DRL2 values in Tunis are so low

Question (2026-09-24): in the manuscript (Tab. B1, Tunis 1980) DRL2 reaches 0.5 t/ha at 300 mm, while DRL1 reaches 3.7 t/ha; above 700 mm DRL2 even loses yield (13.3 down to 12.6 t/ha).

## Answer

The Tunis version of Kelly's code that produced the DRL2 values (repo `jbkze/aquacrop-gym-kelly`, commit a91986f of 2025-09-18) rewards the agent with the **unscaled profit** at the end of the season: 180 USD/t × yield − 1 USD/mm × water − 1,728 USD/ha, i.e. values around −2,000. Kelly's original code divided this profit by 1,000; the commit removed that division. With rewards of this size PPO hardly moves away from its starting policy. In RLlib 1.13 this is made worse by the default clipping of the value loss at 10 (`vf_clip_param`), which cannot follow returns of about −2,000 (not tested here; the effect already appears without it in SB3).

The starting policy proposes about half of the maximum depth at every decision. At 300 mm this means 12–21 mm every 5 days; the budget is used up by day 91, the crop dies before maturity, and yield ends near zero.

The manuscript describes the DRL2 reward as yield at the end of the season (Sec. 3.3.3). The code used profit.

## Evidence

All runs: this repo's environment (checked step by step against the thesis environment), stable-baselines3 PPO with the SB3 equivalents of Kelly's RLlib settings, 100,000 steps, Tunis 1980, three seeds per cell. `drl2_published` uses the years and the decision interval of the code (training 1990–1994, validation 1995–1999, a decision every 5 days).

| Setup | Reward | 300 mm | 600 mm | 800 mm |
|---|---|---|---|---|
| drl2_published | profit, unscaled | 0.65–0.75 | 10.47–13.47 | 13.86–14.04 |
| drl2_published | profit / 1000 | **3.94–4.21** | **13.37–13.63** | not run |
| drl2_published | yield | 3.49–3.99 | 13.38–13.60 | 14.04–14.12 |
| manuscript DRL2 | | 0.5 | 11.9 | 12.6 |
| manuscript DRL1 | | 3.7 | 13.4 | 14.1 |

Test yield in t/ha, range over the three seeds. Runs: `results/runs/2026-09-24_diag_drl2_reward` and `results/runs/2026-09-24_diag_drl2_scale` (configs `config/experiments/diag_drl2_*.yaml` and the method `drl2_published` in `config/agents.yaml`, as of commit e5de53f; both were removed when the plan moved into `config/plan.yaml`).

Irrigation schedules at 300 mm (seed 0): with the unscaled profit reward the agent applies 12–21 mm every 5 days from day 1 to day 91 (crop dies); with the yield reward it applies few 34 mm doses up to day 126 (3.49 t/ha).

## What this does not explain

- Zero yield of both agents at 100–200 mm: DRL1 also fails there. This is the subject of E4.
- The exact published numbers: the published runs used RLlib (not re-run here) and an unknown training length; the notebook saved in the repo stops after about 6,400 steps. The drop above 700 mm fits a single seed per budget: at 600 mm the unscaled-profit runs spread from 10.5 to 13.5 t/ha across seeds.

## Checked and ruled out

- Crop model version (AquaCrop 1.12 vs 3.x): negligible for fixed schedules (manuscript, figure comparing the AquaCrop versions).
- Stale stress observation in AquaCrop 1.1.2: `Ksw` is written to the model state every day.
- The design of DRL2 itself (continuous depth, no mask, reward at the end only): with a yield reward or the scaled profit it reaches the level of DRL1 at 300 and 600 mm.

## Consequence for the revision

The published DRL1 vs DRL2 comparison in Tunis mainly reflects an unscaled reward in the DRL2 code, not the three design choices of DRL1. The manuscript text should state the reward that was actually used, and the claims in the Discussion that attribute the gap to masking, state size and the stepwise reward need the ablation (E2) as evidence.
