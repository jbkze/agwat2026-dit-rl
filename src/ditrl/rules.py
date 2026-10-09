"""Simple scheduling rules under the same seasonal budget as the DRL agents (E1).

Each rule has one free parameter. It is chosen by grid search on the training years, separately
for each site and budget, and then applied unchanged to the test years.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ditrl.env import IrrigationEnv
from ditrl.sites import season_result

Rule = Callable[[IrrigationEnv], float]   # depth in mm for the current decision


def threshold(depletion_fraction: float, max_depth: float) -> Rule:
    """Refill the root zone when its depletion exceeds a fraction of the total available water."""
    def rule(env: IrrigationEnv) -> float:
        s = env.state
        if s.taw > 0 and s.depletion / s.taw >= depletion_fraction:
            return min(float(s.depletion), max_depth)
        return 0.0
    return rule


def fixed_interval(n_doses: int, first_dap: int, last_dap: int) -> Rule:
    """Split the budget into `n_doses` equal doses on evenly spaced days after planting."""
    days = set(np.linspace(first_dap, last_dap, n_doses).round().astype(int).tolist())
    def rule(env: IrrigationEnv) -> float:
        return env.budget / n_doses if int(env.state.dap) in days else 0.0
    return rule


RULES = {"threshold": threshold, "fixed_interval": fixed_interval}


def run_season(env: IrrigationEnv, rule: Rule, year: int) -> dict:
    """One season under a rule: the season result plus the decision log."""
    env.reset(options={"year": year})
    while not env.done:
        env.step_mm(rule(env))
    return {"year": year, **season_result(env.model), "decisions": env.decisions}
