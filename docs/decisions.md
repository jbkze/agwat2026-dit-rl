# Decision log

## 2026-09-24: AquaCrop fork instead of upstream AquaCrop 3.1 with a patch
**Context:** DRL1 observes the leaf-expansion stress coefficient. Upstream AquaCrop-OSPy computes it but does not keep it on the model state; the fork `jbkze/aquacrop-waterstress` does. **Decision:** use the fork (3.0.x), installed from GitHub, the same version as in the paper. **Why:** the published DRL1 results were produced with it, and it is cited in the paper. Moving to upstream 3.1 would change the crop model between the first submission and the revision.

## 2026-09-24: One environment for DRL1, DRL2 and all ablation cells
**Context:** DRL2 ran in Kelly's aquacrop-gym with AquaCrop 1.12 and Ray RLlib 1.13; DRL1 ran in the thesis environment with stable-baselines3. Differences between the two results can come from the design choices, the crop model version and the RL library at the same time. **Decision:** both are configurations of `src/ditrl/env.py`, trained with stable-baselines3. DRL2 gets the SB3 equivalents of Kelly's RLlib settings (method `drl2` in `config/plan.yaml`); the adaptive KL penalty of RLlib has no SB3 counterpart and is left out. **Why:** the ablation (E2) must change one factor at a time; the library and the crop model version have to be the same in every cell. The paper should say that the re-implemented DRL2 can differ from the original numbers for this reason, and report both.

## 2026-09-24: Observation scaling
**Context:** Kelly et al. normalize observations with running mean and standard deviation; the thesis DRL1 uses raw values. **Decision:** DRL1 stays unscaled (as published). DRL2 uses a fixed mean and standard deviation estimated from 20 seasons with random decisions on the training years. In E2 all cells keep the DRL1 setting (unscaled). **Why:** DRL1 should match the published configuration. Open point: with 18 unscaled variables, the observation factor in E2 also contains a scaling effect. If E2 shows a large observation effect, repeat the two KELLY cells with `normalize: true` to separate both.

## 2026-09-24: Oracle with pymoo until the GET-OPTIS code is available
**Context:** GET-OPTIS is not available yet; Niels was asked for the code. **Decision:** a pymoo genetic algorithm with the event-based encoding of GET-OPTIS. A restart stops only when the best yield gains less than 0.001 t/ha over 50 generations (cap of 1,000 generations as a safeguard). Every case gets 9 restarts (3 numbers of events, 3 seeds); a case counts as converged when the two best restarts differ by less than 0.05 t/ha and the best one stopped because of the stall rule. The analysis reads the reference values through `config/plan.yaml` (`analysis.oracle_source`), so GET-OPTIS results replace them by one config change. **Why:** a fixed number of generations stopped the July runs too early (4.73 t/ha after 40 generations against 5.22 t/ha after about 120 generations, Tunis 1980, 300 mm). The number of events matters at low budgets: 10 events reached 1.75 t/ha at 150 mm, 25 events only 0.27 t/ha.

## 2026-09-24: Model selection in Nebraska on the test years
**Decision:** as in the paper and in Kelly et al. (2024), years 71 to 100 select the checkpoint and are also the test years. In Tunis, validation (1997–2001) and test (1980) are separate. **Why:** comparability with the published numbers. The paper names this as a limitation.

## 2026-09-24: Heuristics tuned on the training years
**Decision:** the free parameter of each rule is chosen by grid search on the training years, separately for each site and budget, and then applied unchanged to the test years. **Why:** the DRL agents also see only the training years (plus validation for model selection).

## 2026-09-24: torch 2.4.1 and a compiler check skipped on Windows
**Decision:** torch 2.4.1 instead of 2.4.0 (2.4.0 lacks an OpenMP DLL on Windows). On Windows, `src/ditrl/sites.py` skips numba's check for a C compiler, which the fork triggers at import although nothing is compiled ahead of time in development mode. **Why:** so that the smoke test runs on a laptop without Visual Studio. Linux is unaffected.

## 2026-09-24: Continuous DRL2 action scaled to [-1, 1]
**Context:** in the first check run (`check_paper`, 2026-09-24) the re-implemented DRL2 gave zero yield at 300, 400 and 600 mm after 30,000 steps. The PPO policy of stable-baselines3 starts with mean action 0, which is 0 mm; with a reward only at the end of the season there is no signal to learn from. RLlib 1.13, which Kelly et al. used, rescales actions by default (`normalize_actions: True`), so there an untrained policy proposes about half of the maximum depth. **Decision:** the continuous action lives in [-1, 1] and is mapped linearly to 0..`max_depth`. **Why:** this reproduces the starting behaviour of the original DRL2 and is the recommended setup for continuous actions in stable-baselines3.

## 2026-09-24: 300,000 training steps instead of 100,000
**Context:** the check run `check_paper` (one seed, Tunis 1980) gave DRL1 0.72, 4.84 and 11.00 t/ha at 300, 400 and 600 mm, against 3.7, 7.4 and 13.4 t/ha in the manuscript. A step-by-step comparison with the thesis environment (same actions, Tunis 1980, 400 mm) showed no difference in observations or rewards larger than 2e-6, so the environment is not the cause. The validation curve at 400 mm was still rising steeply at the end (1.57 t/ha after 70,000 steps, 4.37 t/ha after 90,000 steps). **Decision:** every DRL variant trains for 300,000 steps. The learning curves (E6) show whether this is enough. **Why:** a comparison between variants is only fair once each has converged. The higher step count triples the DRL compute to about 9 hours on h200 for all campaigns together.

## 2026-09-24: Stall window of 50 generations, restarts stay essential
**Context:** in `check_oracle` a single restart (10 events, seed 0, Tunis 1980, 300 mm) stopped by the stall rule after 87 generations at 4.04 t/ha; in July, several restarts reached 5.22 t/ha for the same case. The genetic algorithm gets stuck in local optima. **Decision:** stall window 50 generations instead of 30; E0 keeps 9 restarts per case and the convergence flag (two best restarts within 0.05 t/ha). **Why:** a single restart is not a reliable upper bound; agreement between restarts is the evidence that the search has found the best basin it can find.

## 2026-09-24: Note on the manifest of the first E1 run
The manifest of `results/runs/2026-09-24_e1_benchmark` says `git_dirty: true`. The only difference from commit 2ac4e88 was the untracked campaign log `results/run_all.log`, which is ignored since the next commit. The code of that run is exactly commit 2ac4e88.

## 2026-09-24: Cause of the weak published DRL2 values; DRL2 decides every 5 days in Tunis
**Context:** full diagnosis in `docs/diagnosis-drl2.md`. The published Tunis DRL2 runs used an unscaled profit reward (about -2,000); with it the agent barely learns. Scaled by 1/1000 (as in Kelly's original code) or with a yield reward, DRL2 reaches the level of DRL1 at 300 and 600 mm. **Decision:** DRL2 in this repo keeps the yield reward in Tunis (as the manuscript describes it) and decides every 5 days, as the Tunis code did; 7 days (the config default of the code) capped the seasonal water at 19 x 34 mm = 646 mm. Whether the paper keeps the published DRL2 numbers, replaces them, or reports both is a decision for the authors.

## 2026-09-25: Experiments in three stages; published GET-OPTIS values as the oracle
**Context:** the full campaign takes 8 to 9 days on the laptop; how much fits into the revision depends on the time left and on access to h200. **Decision:** the plan is split into three stages (`docs/experiment-plan.md`); stage 1 is the minimum, later stages add seeds, cells, budgets and experiments without repeating jobs. The ablation cells and the low-budget variants are named methods in `config/plan.yaml`, so that each stage trains only new cells. Stages 1 and 2 use the EA-DS and EA-SS values of the manuscript as the oracle (they are GET-OPTIS results), stored in `data/external/`; the own pymoo oracle moves to stage 3. The flat configs `e0` to `e5` are replaced by the staged configs. **Why:** each stage ends with a result set that can go into the paper, and the most expensive part (the oracle) is only needed if the published GET-OPTIS values are to be checked.

## 2026-10-09: No model selection; the last policy is tested
**Context:** the co-author's plan for the ablation forbids any checkpoint or hyperparameter selection on evaluation data. In Nebraska the validation years were the test years (entry of 2026-09-24).
**Decision:** every DRL method, benchmark included, is tested with the policy after its last training step. The validation years only feed the learning curves.
**Why:** a choice made on the test years makes the reported yields optimistic; the same rule for all methods keeps the comparisons fair. This replaces the entry of 2026-09-24 on model selection.

## 2026-10-09: Ablation in steps, with four budget handlings
**Decision:** the ablation starts with the four variants agreed with the co-author and grows to the full 2 x 4 x 2 design in three further parts. Budget handling has four levels: mask, cut, reject (0 mm, no penalty) and penalty (reject, and 1 subtracted from that step's reward).
**Why:** the agreed part is the minimum; the later parts answer the question of interactions without repeating a run.

## 2026-10-09: One plan file, one folder per job
**Decision:** sites, methods, experiments and analysis settings live in `config/plan.yaml`. Every job writes into its own folder `<runs>/jobs/<experiment>/<job id>/`, and a job counts as done when its `result.json` exists. The command line selects stages or experiments and splits them into shards.
**Why:** the job list depends only on the plan, so any number of machines can each run a shard and their result folders merge by copying. A check on 47 reference jobs (all methods, both sites, the oracle) gave results identical to the last digit with the code before the change.

## 2026-10-09: Season starts are copied, not rebuilt
**Context:** building and initializing an AquaCrop model took 35 to 57 ms per season, about 40 % of the environment's time.
**Decision:** the first season of each year is kept as a template; later seasons start from a deep copy (2 ms). Copied and freshly built seasons give identical outputs.
**Why:** about 25 % less time per training. Compiling the fork's solver functions with numba was also tried: identical numbers, but 1.5 to 3.5 times slower, because every call converts large named tuples. A real gain would need the whole daily step in one compiled function, a rewrite of the fork.

## 2026-10-09: No unit tests in this repository
**Decision:** the earlier `tests/test_core.py` was removed at the researcher's request. Correctness of a code change is checked by rerunning reference jobs and comparing them with the previous results.

## 2026-10-09: Result tables in git, raw runs outside
**Decision:** raw job folders (models, logs, about 1 MB per training) stay in `runs/` and in the server archive; `ditrl export` turns them into tables per experiment in `results/<experiment>/`, which are committed and pushed with every batch of results. Every job result records the commit of the code that produced it. Experiment names are descriptive (for example `ablation_one_change` instead of a step letter), and tables and figures name the methods by their design choices.
**Why:** the researcher wants every incoming result in the repository in a structured form; the models of about 3,000 trainings would make the repository several GB large, the tables do not.

## 2026-10-09: Training until convergence, continued from the first round
**Context:** with a fixed 300,000 steps the median validation yield in Tunis still rose by 0.2 to 0.3 t/ha per 100,000 steps; about half of the ablation runs were still rising at the end. In Nebraska the curves were flat.
**Decision:** every DRL run trains until the yield of its training seasons stops rising (blocks of 50,000 steps, gain of more than 0.1 t/ha over the best earlier block, patience 3 blocks, 300,000 to 3,000,000 steps). The 300 runs of the first round are continued from their saved models; their seasons after the resume point come from a seed shifted by the steps already trained.
**Why:** a comparison between variants is only fair once each has converged, and a fixed length favours variants that learn fast. The rule looks at the training seasons, not at validation or test years, because in Nebraska the validation years are the test years. In the first round training and validation levelled off at about the same time, so the training seasons are a usable signal. Continuing instead of restarting saves about 300 x 300,000 steps.


## 2026-10-10: Reporting through the repository
**Decision:** `ditrl report` rebuilds every figure (SciencePlots, PNG and PDF) and table of `reports/`, its index `reports/README.md` and an automatic results block in the main README from the tables in `results/`. It runs after every batch of results. What the results mean is written down by hand, dated, in `reports/findings.md`. The export adds the PPO training statistics (`training.csv.gz`, about every 10,000 steps), and compressed tables carry no time stamp, so unchanged results stay unchanged in git.
**Why:** the researcher wants the results readable in the repository at any time, with the figures and the scripts that make them side by side.
