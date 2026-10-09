"""Sites and the AquaCrop model of one season.

Every method builds its model here, so DRL agents, rules and the oracle solve exactly the same
simulation problem.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from functools import cache

import numba.pycc.platform
import numpy as np
import pandas as pd

from ditrl.paths import REPO

# The AquaCrop fork imports the Python source of its solver functions in DEVELOPMENT mode.
# Without it, the fork expects ahead-of-time compiled modules that it does not ship.
os.environ.setdefault("DEVELOPMENT", "DEVELOPMENT")
# At import time the fork still creates numba's ahead-of-time build objects, which look for a
# C compiler. Nothing is compiled ahead of time in DEVELOPMENT mode, so the check is skipped.
numba.pycc.platform.external_compiler_works = lambda: True

from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement, Soil  # noqa: E402, I001
from aquacrop.utils import get_filepath, prepare_weather  # noqa: E402

PACKAGED_WEATHER = "aquacrop:"  # prefix of weather files shipped with AquaCrop-OSPy
# Irrigation method 5 applies whatever depth the caller sets for the next day; the cap on a single
# application must not bind, because the caller already limits it.
DEPTH_SET_BY_CALLER = 5
NO_SINGLE_DEPTH_CAP_MM = 1000.0
SCHEDULE = 3  # irrigation method: a fixed table of dates and depths


@dataclass(frozen=True)
class Site:
    name: str
    weather: str
    rain: bool
    soil: str
    crop: str
    planting_date: str
    season_end: str
    initial_water_content: dict
    train_years: list[int]
    val_years: list[int]
    test_years: list[int]
    budgets: list[float]

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> Site:
        years = {k: inclusive_range(cfg[k]) for k in ("train_years", "val_years", "test_years")}
        return cls(name=name, **{**cfg, **years})

    def years(self, split: str) -> list[int]:
        """Years of a split: train, val, test, or train_val."""
        if split == "train_val":
            return self.train_years + self.val_years
        return getattr(self, f"{split}_years")


def inclusive_range(bounds: list[int]) -> list[int]:
    """The inclusive range [first, last] of the config (years or seeds) as a list."""
    first, last = bounds
    return list(range(first, last + 1))


@cache
def load_weather(source: str, rain: bool) -> pd.DataFrame:
    """Daily weather, read once per process. Without rain, precipitation is set to 0."""
    if source.startswith(PACKAGED_WEATHER):
        df = prepare_weather(get_filepath(source.removeprefix(PACKAGED_WEATHER)))
    else:
        df = pd.read_csv(REPO / source, parse_dates=["Date"])
        df = df[["MinTemp", "MaxTemp", "Precipitation", "ReferenceET", "Date"]]
    return df if rain else df.assign(Precipitation=0.0)


def new_model(site: Site, year: int, irrigation: IrrigationManagement) -> AquaCropModel:
    """AquaCrop model of one season of `site` in `year`, not started yet."""
    return AquaCropModel(
        sim_start_time=f"{year}/{site.planting_date}",
        sim_end_time=f"{year}/{site.season_end}",
        weather_df=load_weather(site.weather, site.rain),
        soil=Soil(soil_type=site.soil),
        crop=Crop(site.crop, planting_date=site.planting_date),
        initial_water_content=InitialWaterContent(**site.initial_water_content),
        irrigation_management=irrigation,
    )


class SeasonStarts:
    """Started season models whose daily irrigation depth is set by the caller.

    Building and initializing a model takes about 20 times longer than copying a started one, so
    the first season of each year is kept as a template and every later one is a deep copy of it.
    The copy starts from exactly the same state; the weather table is shared, not copied."""

    def __init__(self, site: Site):
        self.site = site
        self._templates: dict[int, AquaCropModel] = {}

    def __call__(self, year: int) -> AquaCropModel:
        if year not in self._templates:
            irrigation = IrrigationManagement(irrigation_method=DEPTH_SET_BY_CALLER,
                                              MaxIrr=NO_SINGLE_DEPTH_CAP_MM)
            model = new_model(self.site, year, irrigation)
            model.run_model(num_steps=1, initialize_model=True)
            self._templates[year] = model
        template = self._templates[year]
        return copy.deepcopy(template, {id(template.weather_df): template.weather_df})


_STARTS: dict[str, SeasonStarts] = {}


def season_starts(site: Site) -> SeasonStarts:
    """The season starts of a site, shared by every environment in this process."""
    if site.name not in _STARTS:
        _STARTS[site.name] = SeasonStarts(site)
    return _STARTS[site.name]


def season_result(model: AquaCropModel) -> dict:
    """Yield, applied water and crop death of a finished season."""
    stats = model._outputs.final_stats
    return {
        "yield_t_ha": float(stats["Dry yield (tonne/ha)"].iloc[0]),
        "irrigation_mm": float(stats["Seasonal irrigation (mm)"].iloc[0]),
        "crop_dead": bool(model._init_cond.crop_dead),
    }


def simulate_schedule(site: Site, year: int, daily_depths: np.ndarray) -> dict:
    """A whole season with a fixed irrigation schedule (mm per day from planting)."""
    dates = pd.date_range(f"{year}/{site.planting_date}", periods=len(daily_depths))
    schedule = pd.DataFrame({"Date": dates, "Depth": np.asarray(daily_depths, float)})
    irrigation = IrrigationManagement(irrigation_method=SCHEDULE, Schedule=schedule)
    model = new_model(site, year, irrigation)
    model.run_model(till_termination=True)
    return season_result(model)
