# Central findings

Written in `config/findings.md`; the numbers are filled in from the current results by `ditrl report`.

**1. Rewarding only the yield at the end works better than the stepwise reward of DRL1.** Changing only the reward raises test yield by +1.01 t/ha [+0.55, +1.42] in Nebraska at 75 mm and by +2.39 t/ha [+1.54, +3.27] in Tunis at 300 mm. The stepwise reward pays relative evapotranspiration every day, which is cheapest to earn early in the season while the canopy is small, so DRL1 spends its water before the stages that decide the yield. In Tunis at 150 and 200 mm the crop has so far died in every run with the stepwise reward, whatever the other choices. Only with a large budget (Tunis 500 mm, +0.08 t/ha [-0.30, +0.57]) is the timing nearly the same and so is the yield.

![agreed_irrigation_timing](figures/agreed_irrigation_timing.png)

**2. The 18 variables of Kelly et al. help in Nebraska and hurt in Tunis.** In Nebraska the 18 variables raise test yield by +1.05 t/ha [+0.65, +1.43] at 75 mm, in Tunis they lower it by -1.47 t/ha [-2.14, -0.56] at 500 mm. Nebraska has rain and weather that differs from year to year, which the extra variables can describe. Tunis is simulated without rain, so there they plausibly add mostly noise. The 18 variables enter unscaled, and the run with scaled variables, still pending, shows whether part of the effect is a scaling effect.

**3. How the budget is enforced matters less than the reward, and its effect depends on the case.** Rejecting a dose that exceeds the remaining budget instead of masking it changes test yield by -0.15 t/ha [-0.80, +0.50] in Nebraska at 75 mm and by +0.19 t/ha [-0.82, +1.26] in Tunis at 300 mm. Adding a penalty to the rejection helps in Tunis at 300 mm (+1.81 t/ha [+0.95, +2.62]) but costs yield at 600 mm (-1.20 t/ha [-1.95, -0.46, 8 of 10 runs]). Masking, the choice of DRL1, is therefore not better than the alternatives. Where a single change and the same change averaged over the other choices (diamonds in the figure below) differ, the choices interact, most clearly for the reward in Tunis at 500 mm.

**4. Training until convergence instead of 300,000 steps changed the yield little, except in Tunis at 300 mm.** Converged runs and the same runs after 300,000 steps differ by +0.17 t/ha [-0.28, +0.75] in Nebraska at 75 mm and by -0.46 t/ha [-1.02, -0.03] in Tunis at 300 mm (300,000 steps minus converged), so the earlier results were not an artefact of training too short.
