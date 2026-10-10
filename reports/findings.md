# Findings

Notable results, written down when a batch of results comes in; newest first. The figures are rebuilt by `ditrl report` and always show the current data; the numbers quoted here are those of the date in the heading.

## 2026-10-10: Agreed ablation and full design, trained until convergence

**The reward is the strongest design choice.** Rewarding the yield at the end of the season only, instead of relative evapotranspiration at every step plus the yield, raises test yield in every case: +0.45 t/ha in Nebraska at 75 mm, +1.84 t/ha in Tunis at 300 mm and +0.81 t/ha at 500 mm (main effects over the full 2 x 4 x 2 design, 10 seeds, all significant after Holm correction). The extra budgets of the ablation, which are still coming in, show the same direction up to 400 mm.
![Main effects](figures/ablation_main_effects.png)

**DRL1 is in no case the best of the four agreed variants.** With 10 seeds: Nebraska 75 mm DRL1 10.95 t/ha against 12.00 with 18 variables and 11.97 with the yield at the end; Tunis 300 mm 1.82 against 4.22 with the yield at the end; Tunis 500 mm 9.70 against 9.78. [Results table](tables/agreed_ablation_results_10_seeds.md)
![Agreed ablation, 10 seeds](figures/agreed_ablation_10_seeds.png)

**Masking pays off in Nebraska only.** Every other budget handling costs 0.06 to 0.25 t/ha there, while at Tunis 300 mm cutting to the remaining budget and rejecting with a penalty are 0.72 and 0.87 t/ha better than masking.

**DRL1 and reject do not settle in Nebraska.** Their validation yield rises to about 11.2 t/ha around 400,000 steps and falls again towards the end of training, with a wide spread over seeds; the variants with 18 variables or the yield at the end climb higher and stay there. Testing the last policy, as agreed, therefore costs DRL1 more than the other variants.
![Learning curves](figures/agreed_learning_curves.png)

**With 18 variables the policy updates keep growing.** The approximate KL divergence between successive policies rises steadily over training with 18 variables but stays flat with the 5 variables of DRL1. The 18 variables enter unscaled (for example biomass and the day of the month next to fractions), which plausibly makes the updates harder to control; the experiment with scaled observations (`obs_scaled`, stage 3) tests this. With the yield at the end as the only reward, the value function explains much less of the return, as expected for a sparse reward, without costing yield.
![Training dynamics](figures/agreed_training_dynamics.png)

**Training until convergence changed little.** Continuing the 300,000-step runs until the stopping rule ended them added a median 200,000 steps and raised test yield by 0.14 to 0.26 t/ha on average; about a quarter of the runs ended slightly lower, and the ranking of the variants stayed the same.
![Convergence gain](figures/convergence_gain.png)
