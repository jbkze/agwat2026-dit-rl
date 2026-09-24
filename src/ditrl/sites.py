"""Site setup: weather, soil, crop and the AquaCrop model for one season.

All methods build their model through `new_model`, so DRL agents, heuristics and the
oracle solve exactly the same simulation problem.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("DEVELOPMENT", "DEVELOPMENT")  # the fork runs in plain Python/numba JIT mode
if sys.platform == "win32":
    # The fork creates numba AOT build objects at import time, which checks for a C compiler.
    # In JIT mode nothing is ever compiled ahead of time, so the check is skipped on Windows
    # machines without MSVC. Linux (h200) has gcc and does not need this.
    import numba.pycc.platform as _pycc_platform
    _pycc_platform.external_compiler_works = lambda: True

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", FutureWarning)

from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement, Soil  # noqa: E402
from aquacrop.utils import get_filepath, prepare_weather  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def year_range(bounds: list[int]) -> list[int]:
    """Expand an inclusive range [first, last] from the config into a list of years."""
    first, last = bounds
    return list(range(first, last + 1))


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
        years = {k: year_range(cfg[k]) for k in ("train_years", "val_years", "test_years")}
        rest = {k: v for k, v in cfg.items() if k not in years}
        return cls(name=name, **rest, **years)


@lru_cache(maxsize=None)
def load_weather(source: str, rain: bool) -> pd.DataFrame:
    """Read the daily weather once per process. Without rain, precipitation is set to 0."""
    if source.startswith("aquacrop:"):
        df = prepare_weather(get_filepath(source.removeprefix("aquacrop:")))
    else:
        df = pd.read_csv(REPO / source)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[["MinTemp", "MaxTemp", "Precipitation", "ReferenceET", "Date"]]
    if not rain:
        df = df.assign(Precipitation=0.0)
    return df


def new_model(site: Site, year: int, irrigation: IrrigationManagement) -> AquaCropModel:
    """Create an AquaCrop model for one season of `site` in `year` (not yet started)."""
    return AquaCropModel(
        sim_start_time=f"{year}/{site.planting_date}",
        sim_end_time=f"{year}/{site.season_end}",
        weather_df=load_weather(site.weather, site.rain),
        soil=Soil(soil_type=site.soil),
        crop=Crop(site.crop, planting_date=site.planting_date),
        initial_water_content=InitialWaterContent(**site.initial_water_content),
        irrigation_management=irrigation,
    )


def season_result(model: AquaCropModel) -> dict:
    """Yield, applied water and crop death of a finished season."""
    stats = model._outputs.final_stats
    return {
        "yield_t_ha": float(stats["Dry yield (tonne/ha)"].iloc[0]),
        "irrigation_mm": float(stats["Seasonal irrigation (mm)"].iloc[0]),
        "crop_dead": bool(model._init_cond.crop_dead),
    }


def simulate_schedule(site: Site, year: int, daily_depths: np.ndarray) -> dict:
    """Run a whole season with a fixed daily irrigation schedule (mm per day from planting)."""
    dates = pd.date_range(f"{year}/{site.planting_date}", periods=len(daily_depths))
    schedule = pd.DataFrame({"Date": dates, "Depth": np.asarray(daily_depths, float)})
    model = new_model(site, year, IrrigationManagement(irrigation_method=3, Schedule=schedule))
    model.run_model(till_termination=True)
    return season_result(model)


def season_length(site: Site, year: int) -> int:
    """Number of simulated days from planting to the configured season end."""
    start = pd.Timestamp(f"{year}/{site.planting_date}")
    end = pd.Timestamp(f"{year}/{site.season_end}")
    return (end - start).days + 1
