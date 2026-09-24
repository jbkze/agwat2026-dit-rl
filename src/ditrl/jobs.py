"""What one job does. Every handler writes its files into the job folder `out` and returns a
small summary dict that the runner stores as `result.json`."""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ditrl import drl, heuristics, oracle
from ditrl.config import load_sites
from ditrl.env import EnvSpec, IrrigationEnv
from ditrl.sites import REPO, year_range

log = logging.getLogger(__name__)


def env_spec(job: dict, years: list[int]) -> EnvSpec:
    """Environment settings of a DRL or heuristic job."""
    cfg, info = job["config"], load_sites()
    return EnvSpec(site=info["sites"][job["site"]], budget=job["budget"], years=list(years),
                   observation=cfg.get("observation", "DRL1"), depths=cfg.get("depths"),
                   max_depth=cfg.get("max_depth", 1000.0), interval=cfg["interval"],
                   constraint=cfg.get("constraint", "clip"), reward=cfg.get("reward", "terminal"),
                   reward_scale=cfg.get("reward_scale", 1.0), prices=info["prices"])


def _year_split(job: dict) -> tuple[list[int], list[int], list[int]]:
    site = load_sites()["sites"][job["site"]]
    if job.get("train_years"):  # E3: train, select and test on the same single year
        return job["train_years"], job["train_years"], job["train_years"]
    years = job["config"].get("years", {})  # a method may use its own split, e.g. Kelly's code
    train = year_range(years["train"]) if "train" in years else site.train_years
    val = year_range(years["val"]) if "val" in years else site.val_years
    return train, val, site.test_years


def _summary(seasons: list[dict]) -> dict:
    return {"years": [s["year"] for s in seasons],
            "yield_t_ha": [s["yield_t_ha"] for s in seasons],
            "irrigation_mm": [s["irrigation_mm"] for s in seasons],
            "crop_dead": [s["crop_dead"] for s in seasons],
            "mean_yield_t_ha": float(np.mean([s["yield_t_ha"] for s in seasons]))}


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=float))


def run_drl(job: dict, out: Path) -> dict:
    train_years, val_years, test_years = _year_split(job)
    spec = env_spec(job, train_years)
    if job["config"].get("normalize"):
        spec = replace(spec, scale=drl.observation_scale(spec, seasons=20, seed=job["seed"]))
        _dump(out / "scale.json", spec.scale)
    model = drl.train(spec, replace(spec, years=val_years), job["config"], job["seed"], out)
    test = drl.evaluate(model, spec, test_years)
    val = drl.evaluate(model, spec, val_years)
    _dump(out / "seasons.json", {"test": test, "val": val})
    return {"test": _summary(test), "val": _summary(val)}


def run_heuristic(job: dict, out: Path) -> dict:
    """Grid search of the free parameter on the training years, then the test years."""
    cfg = job["config"]
    train_years, _, test_years = _year_split(job)
    env = IrrigationEnv(env_spec(job, train_years))
    grid = cfg.get("grid", {})
    candidates = [dict(zip(grid, values)) for values in itertools.product(*grid.values())] or [{}]
    scores = []
    for cand in candidates:
        rule = heuristics.RULES[cfg["rule"]](**cfg.get("params", {}), **cand)
        mean = float(np.mean([heuristics.run_season(env, rule, y)["yield_t_ha"]
                              for y in train_years]))
        scores.append({**cand, "train_mean_yield": mean})
        log.info("%s: mean training yield %.3f t/ha", cand, mean)
    best = max(scores, key=lambda s: s["train_mean_yield"])
    params = {k: best[k] for k in grid}
    rule = heuristics.RULES[cfg["rule"]](**cfg.get("params", {}), **params)
    test = [heuristics.run_season(env, rule, y) for y in test_years]
    _dump(out / "grid.json", scores)
    _dump(out / "seasons.json", {"test": test})
    return {"params": params, "test": _summary(test)}


def run_oracle(job: dict, out: Path) -> dict:
    site = load_sites()["sites"][job["site"]]
    res = oracle.optimize(site, job["years"], float(job["budget"]), n_events=job["n_events"],
                          seed=job["seed"], window=job["window"], quantile=job["quantile"],
                          **job["ga"])
    _dump(out / "history.json", res.history)
    log.info("objective %.3f t/ha after %d generations (%s)", res.objective, res.generations,
             res.stopped_by)
    result = asdict(res)
    result.pop("history")
    return result


def run_evaluate(job: dict, out: Path) -> dict:
    """E5: a trained model on the test years with noisy observations."""
    src_dir, src = REPO / job["source"], job["source_job"]
    _, _, test_years = _year_split(src)
    spec = replace(env_spec(src, test_years), noise=job["noise"])
    if (src_dir / "scale.json").exists():
        spec = replace(spec, scale=json.loads((src_dir / "scale.json").read_text()))
    algo = drl.MaskablePPO if spec.constraint == "mask" else drl.PPO
    model = algo.load(src_dir / "best_model.zip", device="cpu")
    test = drl.evaluate(model, spec, test_years, seed=job["seed"])
    _dump(out / "seasons.json", {"test": test})
    return {"test": _summary(test)}


HANDLERS = {"drl": run_drl, "heuristic": run_heuristic, "oracle": run_oracle,
            "evaluate": run_evaluate}
