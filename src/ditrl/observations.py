"""Observation features. Each feature reads one or more numbers from the running model.

An observation set in `config/agents.yaml` is a list of feature names from `FEATURES`. The two
sets used in the paper are `DRL1` (5 variables) and `KELLY` (the 18 variables of Kelly et al.
2024, in the order of their code). `FORECAST` adds a perfect 7-day weather forecast.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ditrl.env import IrrigationEnv

# Column indices of the daily output tables of the AquaCrop fork.
FLUX = {"Wr": 3, "DeepPerc": 9, "Es": 12, "EsPot": 13, "Tr": 14, "TrPot": 15}
PRECIP, ETO = 2, 3  # columns of the model's weather array


def flux(env: IrrigationEnv, name: str) -> float:
    """Value of a water flux on the last simulated day."""
    row = env.model._clock_struct.time_step_counter - 1
    return float(env.model._outputs.water_flux[row, FLUX[name]])


def _weather(env: IrrigationEnv, start: int, end: int) -> np.ndarray:
    """Precipitation and reference ET for simulation days [start, end), clipped to the season."""
    t0 = env.model._clock_struct.time_step_counter
    lo, hi = max(0, t0 + start), max(0, t0 + end)
    return env.model._weather[lo:hi, PRECIP : ETO + 1].astype(float)


def _mean(block: np.ndarray) -> list[float]:
    return block.mean(axis=0).tolist() if len(block) else [0.0, 0.0]


def _growth_stage_one_hot(env: IrrigationEnv) -> list[float]:
    stage = int(np.clip(int(env.state.growth_stage) - 1, 0, 3))
    return [float(i == stage) for i in range(4)]


def _relative_depletion(env: IrrigationEnv) -> float:
    taw = env.state.taw
    return float(env.state.depletion / taw) if taw > 0 else 0.0


def _date(env: IrrigationEnv):
    return env.model._clock_struct.time_span[env.model._clock_struct.time_step_counter]


# name -> function returning a list of floats
FEATURES: dict[str, Callable[[IrrigationEnv], list[float]]] = {
    # DRL1
    "remaining_budget": lambda e: [e.remaining],
    "root_zone_water": lambda e: [flux(e, "Wr")],
    "deep_percolation": lambda e: [flux(e, "DeepPerc")],
    "expansion_stress": lambda e: [float(e.state.exp)],
    "growth_stage": lambda e: [float(e.state.growth_stage)],
    # Kelly et al. (2024)
    "day_of_month": lambda e: [float(_date(e).day)],
    "month": lambda e: [float(_date(e).month)],
    "relative_depletion": lambda e: [_relative_depletion(e)],
    "days_after_planting": lambda e: [float(e.state.dap)],
    "irrigation_so_far": lambda e: [float(e.state.irr_cum)],
    "canopy_cover": lambda e: [float(e.state.canopy_cover)],
    "biomass": lambda e: [float(e.state.biomass)],
    "weather_last_7_days": lambda e: _mean(_weather(e, -7, 0)),
    "weather_season_sum": lambda e: _weather(e, -int(e.state.dap), 0).sum(axis=0).tolist(),
    "weather_yesterday": lambda e: _mean(_weather(e, -1, 0)),
    "growth_stage_one_hot": _growth_stage_one_hot,
    # perfect forecast (E3)
    "weather_next_7_days": lambda e: _mean(_weather(e, 0, 7)),
}

SETS = {
    "DRL1": ["remaining_budget", "root_zone_water", "deep_percolation",
             "expansion_stress", "growth_stage"],
    "KELLY": ["day_of_month", "month", "relative_depletion", "days_after_planting",
              "irrigation_so_far", "canopy_cover", "biomass", "remaining_budget",
              "weather_last_7_days", "weather_season_sum", "weather_yesterday",
              "growth_stage_one_hot"],
}
SETS["DRL1_FORECAST"] = SETS["DRL1"] + ["weather_next_7_days"]


def resolve(spec: str | list[str]) -> list[str]:
    """Turn a set name or an explicit feature list into a feature list."""
    names = SETS[spec] if isinstance(spec, str) else list(spec)
    unknown = [n for n in names if n not in FEATURES]
    if unknown:
        raise KeyError(f"unknown observation features: {unknown}")
    return names


def width(names: list[str]) -> int:
    """Length of the observation vector: weather features have 2 values, the one-hot stage 4."""
    return sum(2 if n.startswith("weather_") else 4 if n.endswith("one_hot") else 1 for n in names)


def observe(env: IrrigationEnv, names: list[str], noise: dict[str, float],
            rng: np.random.Generator) -> np.ndarray:
    """Observation vector. `noise` maps a feature to the relative standard deviation of
    Gaussian noise on its value (E5); features not listed stay exact."""
    values = []
    for n in names:
        v = FEATURES[n](env)
        if n in noise:
            v = [x * (1.0 + noise[n] * rng.standard_normal()) for x in v]
        values.extend(v)
    return np.array(values, dtype=np.float32)
