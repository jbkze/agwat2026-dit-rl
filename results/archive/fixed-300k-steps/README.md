# Ablation with a fixed training length of 300,000 steps (2026-10-09)

These tables come from the first round of the ablation, in which every DRL run trained for exactly 300,000 environment steps and was tested with its final policy. The learning curves showed that the Tunis runs had not converged by then (the median validation yield at 500 mm still rose by about 0.3 t/ha per 100,000 steps), so the plan was changed to train every run until it converges and all runs were repeated. These tables are kept for comparison; the current results are in `results/<experiment>/`.

Contents: the agreed ablation (60 runs), budget handling (45), interactions (135) and the first quarter of the extra seeds (60), with their run manifests. The layout is the one described in the main README.
