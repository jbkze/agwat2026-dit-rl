"""Open-loop schedule optimization with a genetic algorithm (pymoo), experiment E0.

EA-DS: maximize the yield of one known weather year (perfect foresight).
EA-SS: maximize the 10 % quantile of yield over many years (one robust schedule).

A schedule is encoded as a fixed number of irrigation events, each with a day and a depth.
This encoding follows GET-OPTIS (Schütze et al. 2011). A dense vector with one depth per day
does not work on sand: small daily doses percolate before the roots take them up, so almost
every random schedule gives zero yield and the search has no gradient to follow.

The budget is enforced while decoding: if the depths add up to more than the budget, all of
them are scaled down by the same factor. Every evaluated schedule is therefore feasible.

One call of `optimize` is one restart. The runner starts several restarts per case (different
numbers of events and seeds); `analysis/collect.py` keeps the best one and marks the case as
converged only if the two best restarts agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.termination import Termination
from pymoo.optimize import minimize

from ditrl.sites import Site, simulate_schedule


def decode(x: np.ndarray, n_events: int, window: int, budget: float) -> np.ndarray:
    """Turn [days of all events, depths of all events] into daily depths within the budget.
    Events on the same day add up."""
    days = np.clip(np.round(x[:n_events]).astype(int), 0, window - 1)
    daily = np.zeros(window)
    np.add.at(daily, days, np.clip(x[n_events:], 0.0, None))
    total = daily.sum()
    if total > budget:
        daily *= budget / total
    return daily


class ScheduleProblem(ElementwiseProblem):
    """Minimize minus the yield (EA-DS) or minus the yield quantile over `years` (EA-SS)."""

    def __init__(self, site: Site, years: list[int], budget: float, n_events: int,
                 window: int, max_event_mm: float, quantile: float | None):
        self.site, self.years, self.budget = site, years, budget
        self.n_events, self.window, self.quantile = n_events, window, quantile
        lower = np.zeros(2 * n_events)
        upper = np.concatenate([np.full(n_events, window - 1e-3), np.full(n_events, max_event_mm)])
        super().__init__(n_var=2 * n_events, n_obj=1, xl=lower, xu=upper)

    def season_yields(self, x) -> np.ndarray:
        daily = decode(x, self.n_events, self.window, self.budget)
        return np.array([simulate_schedule(self.site, y, daily)["yield_t_ha"] for y in self.years])

    def _evaluate(self, x, out, *args, **kwargs):
        yields = self.season_yields(x)
        score = yields[0] if self.quantile is None else np.quantile(yields, self.quantile)
        out["F"] = [-score]


class StallTermination(Termination):
    """Stop when the best value has improved by less than `tol` over the last `period`
    generations. `max_gen` is a safeguard only; hitting it is recorded as not converged."""

    def __init__(self, tol: float, period: int, max_gen: int):
        super().__init__()
        self.tol, self.period, self.max_gen = tol, period, max_gen
        self.best: list[float] = []
        self.reason = ""

    def _update(self, algorithm) -> float:
        self.best.append(float(algorithm.opt.get("F").min()))
        if algorithm.n_gen >= self.max_gen:
            self.reason = "max_gen"
            return 1.0
        if len(self.best) > self.period and self.best[-self.period - 1] - self.best[-1] < self.tol:
            self.reason = "stall"
            return 1.0
        return 0.0


class History(Callback):
    """Best value and number of evaluations after every generation (convergence curve)."""

    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []

    def notify(self, algorithm):
        self.rows.append({"generation": algorithm.n_gen, "evaluations": algorithm.evaluator.n_eval,
                          "best": -float(algorithm.opt.get("F").min())})


def concentrated_start(pop_size: int, n_events: int, window: int, max_event_mm: float,
                       budget: float, rng: np.random.Generator, share: float) -> np.ndarray:
    """Initial population. A `share` of it gets 2 to 8 active events of similar size, the
    rest is uniform random. On sand the best schedules use a few large doses, which a purely
    random start rarely finds when `n_events` is large."""
    upper = np.concatenate([np.full(n_events, window - 1e-3), np.full(n_events, max_event_mm)])
    pop = rng.uniform(0.0, upper, size=(pop_size, 2 * n_events))
    for i in range(int(round(pop_size * share))):
        k = int(rng.integers(2, min(n_events, 8) + 1))
        depths = np.zeros(n_events)
        active = rng.choice(n_events, size=k, replace=False)
        depths[active] = min(max_event_mm, budget / k) * rng.uniform(0.6, 1.0, size=k)
        pop[i] = np.concatenate([rng.uniform(5, max(6, window - 20), size=n_events), depths])
    return pop


@dataclass
class OracleResult:
    objective: float           # yield (EA-DS) or yield quantile (EA-SS), t/ha
    yields: list[float]        # yield per year of the best schedule
    schedule: list[float]      # daily depths in mm
    used_mm: float
    generations: int
    evaluations: int
    stopped_by: str            # "stall" or "max_gen"
    history: list[dict]


def optimize(site: Site, years: list[int], budget: float, *, n_events: int, seed: int,
             window: int, max_event_mm: float, pop_size: int, tol: float, period: int,
             max_gen: int, seeded_share: float, quantile: float | None = None) -> OracleResult:
    """One GA restart for one case (site, years, budget)."""
    problem = ScheduleProblem(site, years, budget, n_events, window, max_event_mm, quantile)
    start = concentrated_start(pop_size, n_events, window, max_event_mm, budget,
                               np.random.default_rng(seed), seeded_share)
    history = History()
    res = minimize(problem, GA(pop_size=pop_size, sampling=start, eliminate_duplicates=True),
                   StallTermination(tol, period, max_gen), seed=seed, callback=history,
                   verbose=False, copy_termination=False)
    termination = res.algorithm.termination
    daily = decode(res.X, n_events, window, budget)
    return OracleResult(
        objective=-float(res.F[0]), yields=problem.season_yields(res.X).tolist(),
        schedule=np.round(daily, 3).tolist(), used_mm=float(daily.sum()),
        generations=len(history.rows), evaluations=int(res.algorithm.evaluator.n_eval),
        stopped_by=termination.reason, history=history.rows)
