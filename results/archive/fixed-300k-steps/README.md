# Ablation with a fixed training length of 300,000 steps (2026-10-09)

These tables come from the first round of the ablation, in which every DRL run trained for exactly 300,000 environment steps and was tested with its final policy. Because the Tunis runs were still improving at that point (the median validation yield at 500 mm rose by about 0.3 t/ha per 100,000 steps), all runs were trained further from their saved models until the yield of their training seasons stopped rising. These tables are kept for comparison, and the current results are in `results/<experiment>/`.

Contents: the agreed ablation (60 runs), budget handling (45 runs), interactions (135 runs) and the first quarter of the extra seeds (60 runs), with their run manifests. The folder layout is described in the main README.
