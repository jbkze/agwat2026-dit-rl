# Decision log

## 2026-09-24: AquaCrop fork instead of upstream AquaCrop 3.1 with a patch
**Context:** DRL1 observes the leaf-expansion stress coefficient. Upstream AquaCrop-OSPy computes
it but does not keep it on the model state; the fork `jbkze/aquacrop-waterstress` does.
**Decision:** use the fork (3.0.x), installed from GitHub, the same version as in the paper.
**Why:** the published DRL1 results were produced with it, and it is cited in the paper.
Moving to upstream 3.1 would change the crop model between the first submission and the revision.

## 2026-09-24: One environment for DRL1, DRL2 and all ablation cells
**Context:** DRL2 ran in Kelly's aquacrop-gym with AquaCrop 1.12 and Ray RLlib 1.13; DRL1 ran
in the thesis environment with stable-baselines3. Differences between the two results can come
from the design choices, the crop model version and the RL library at the same time.
**Decision:** both are configurations of `src/ditrl/env.py`, trained with stable-baselines3.
DRL2 gets the SB3 equivalents of Kelly's RLlib settings (`config/agents.yaml`); the adaptive KL
penalty of RLlib has no SB3 counterpart and is left out.
**Why:** the ablation (E2) must change one factor at a time; the library and the crop model
version have to be the same in every cell. The paper should say that the re-implemented DRL2
can differ from the original numbers for this reason, and report both.

## 2026-09-24: Observation scaling
**Context:** Kelly et al. normalize observations with running mean and standard deviation; the
thesis DRL1 uses raw values.
**Decision:** DRL1 stays unscaled (as published). DRL2 uses a fixed mean and standard deviation
estimated from 20 seasons with random decisions on the training years. In E2 all cells keep the
DRL1 setting (unscaled).
**Why:** DRL1 should match the published configuration. Open point: with 18 unscaled variables,
the observation factor in E2 also contains a scaling effect. If E2 shows a large observation
effect, repeat the two KELLY cells with `normalize: true` to separate both.

## 2026-09-24: Oracle with pymoo until the GET-OPTIS code is available
**Context:** GET-OPTIS is not available yet; Niels was asked for the code.
**Decision:** a pymoo genetic algorithm with the event-based encoding of GET-OPTIS. A restart
stops only when the best yield gains less than 0.001 t/ha over 50 generations (cap of 1,000
generations as a safeguard). Every case gets 9 restarts (3 numbers of events, 3 seeds); a case
counts as converged when the two best restarts differ by less than 0.05 t/ha and the best one
stopped because of the stall rule. The analysis reads the reference values through
`config/analysis.yaml`, so GET-OPTIS results replace them by one config change.
**Why:** a fixed number of generations stopped the July runs too early (4.73 t/ha after
40 generations against 5.22 t/ha after about 120 generations, Tunis 1980, 300 mm). The number
of events matters at low budgets: 10 events reached 1.75 t/ha at 150 mm, 25 events only 0.27 t/ha.

## 2026-09-24: Model selection in Nebraska on the test years
**Decision:** as in the paper and in Kelly et al. (2024), years 71 to 100 select the checkpoint
and are also the test years. In Tunis, validation (1997–2001) and test (1980) are separate.
**Why:** comparability with the published numbers. The paper names this as a limitation.

## 2026-09-24: Heuristics tuned on the training years
**Decision:** the free parameter of each rule is chosen by grid search on the training years,
separately for each site and budget, and then applied unchanged to the test years.
**Why:** the DRL agents also see only the training years (plus validation for model selection).

## 2026-09-24: torch 2.4.1 and a compiler check skipped on Windows
**Decision:** torch 2.4.1 instead of 2.4.0 (2.4.0 lacks an OpenMP DLL on Windows). On Windows,
`src/ditrl/sites.py` skips numba's check for a C compiler, which the fork triggers at import
although nothing is compiled ahead of time in development mode.
**Why:** so that the smoke test runs on a laptop without Visual Studio. Linux is unaffected.

## 2026-09-24: Continuous DRL2 action scaled to [-1, 1]
**Context:** in the first check run (`check_paper`, 2026-09-24) the re-implemented DRL2 gave zero
yield at 300, 400 and 600 mm after 30,000 steps. The PPO policy of stable-baselines3 starts with
mean action 0, which is 0 mm; with a reward only at the end of the season there is no signal to
learn from. RLlib 1.13, which Kelly et al. used, rescales actions by default
(`normalize_actions: True`), so there an untrained policy proposes about half of the maximum depth.
**Decision:** the continuous action lives in [-1, 1] and is mapped linearly to 0..`max_depth`.
**Why:** this reproduces the starting behaviour of the original DRL2 and is the recommended
setup for continuous actions in stable-baselines3.

## 2026-09-24: 300,000 training steps instead of 100,000
**Context:** the check run `check_paper` (one seed, Tunis 1980) gave DRL1 0.72, 4.84 and
11.00 t/ha at 300, 400 and 600 mm, against 3.7, 7.4 and 13.4 t/ha in the manuscript. A
step-by-step comparison with the thesis environment (same actions, Tunis 1980, 400 mm) showed no
difference in observations or rewards larger than 2e-6, so the environment is not the cause. The
validation curve at 400 mm was still rising steeply at the end (1.57 t/ha after 70,000 steps,
4.37 t/ha after 90,000 steps).
**Decision:** every DRL variant trains for 300,000 steps. The learning curves (E6) show whether
this is enough.
**Why:** a comparison between variants is only fair once each has converged. The higher step
count triples the DRL compute to about 9 hours on h200 for all campaigns together.

## 2026-09-24: Stall window of 50 generations, restarts stay essential
**Context:** in `check_oracle` a single restart (10 events, seed 0, Tunis 1980, 300 mm) stopped
by the stall rule after 87 generations at 4.04 t/ha; in July, several restarts reached 5.22 t/ha
for the same case. The genetic algorithm gets stuck in local optima.
**Decision:** stall window 50 generations instead of 30; E0 keeps 9 restarts per case and the
convergence flag (two best restarts within 0.05 t/ha).
**Why:** a single restart is not a reliable upper bound; agreement between restarts is the
evidence that the search has found the best basin it can find.

## 2026-09-24: Note on the manifest of the first E1 run
The manifest of `results/runs/2026-09-24_e1_benchmark` says `git_dirty: true`. The only
difference from commit 2ac4e88 was the untracked campaign log `results/run_all.log`, which is
ignored since the next commit. The code of that run is exactly commit 2ac4e88.
