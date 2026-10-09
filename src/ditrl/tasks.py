"""What one job does. Each task writes its files into the job folder and returns a small summary,
which the runner stores as `result.json`.

Files of a job folder (besides job.json, job.log and result.json):

    seasons.csv          one row per evaluated season: split, year, yield, water, crop death
    decisions.csv        one row per decision in the test seasons: day, budget, depths, reward
    learning.csv         DRL: validation yield over training steps
    model.zip            DRL: the trained policy; hyperparameters.json, progress.csv and
                         training.monitor.csv from stable-baselines3; scale.json if normalized
    grid.csv             rules: training yield of every parameter value
    history.csv          oracle: best value per generation
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ditrl import drl, oracle, rules
from ditrl.env import EnvSpec, IrrigationEnv
from ditrl.plan import Job, Plan
from ditrl.store import Store, read_json, write_csv, write_json


def env_spec(job: Job, plan: Plan, years: list[int]) -> EnvSpec:
    """Environment of a DRL, rule or evaluation job."""
    cfg = job.config
    return EnvSpec(site=plan.sites[job.site], budget=job.budget, years=list(years),
                   observation=cfg.get("observation", "DRL1"), interval=cfg["interval"],
                   constraint=cfg.get("constraint", "clip"), reward=cfg.get("reward", "terminal"),
                   depths=cfg.get("depths"), max_depth=cfg.get("max_depth"),
                   penalty=cfg.get("penalty", 0.0), prices=plan.prices,
                   noise=cfg.get("noise", {}))


def year_split(job: Job, plan: Plan) -> tuple[list[int], list[int], list[int]]:
    """Training, validation and test years of a job."""
    if job.train_years:   # E3: train and test on the same single year
        return job.train_years, job.train_years, job.train_years
    site = plan.sites[job.site]
    return site.train_years, site.val_years, site.test_years


def run_drl(job: Job, plan: Plan, out: Path, store: Store) -> dict:
    train_years, val_years, test_years = year_split(job, plan)
    spec = env_spec(job, plan, train_years)
    if job.config.get("normalize"):
        spec = replace(spec, scale=drl.observation_scale(spec, job.config["normalize_seasons"],
                                                         job.seed))
        write_json(out / "scale.json", spec.scale)
    model, curve = drl.train(spec, replace(spec, years=val_years), job.config, job.seed, out)
    write_csv(out / "learning.csv", curve)
    seasons = {"test": drl.evaluate(model, spec, test_years),
               "val": drl.evaluate(model, spec, val_years)}
    write_seasons(out, seasons)
    return summary(seasons)


def run_rule(job: Job, plan: Plan, out: Path, store: Store) -> dict:
    """Grid search of the free parameter on the training years, then the test years."""
    cfg = job.config
    train_years, _, test_years = year_split(job, plan)
    env = IrrigationEnv(env_spec(job, plan, train_years))
    grid = cfg.get("grid", {})
    candidates = [dict(zip(grid, values, strict=True))
                  for values in itertools.product(*grid.values())]
    scores = []
    for candidate in candidates:
        rule = rules.RULES[cfg["rule"]](**cfg["params"], **candidate)
        yields = [rules.run_season(env, rule, y)["yield_t_ha"] for y in train_years]
        scores.append({**candidate, "train_mean_yield": float(np.mean(yields))})
    write_csv(out / "grid.csv", scores)
    best = max(scores, key=lambda s: s["train_mean_yield"])
    params = {k: best[k] for k in grid}
    rule = rules.RULES[cfg["rule"]](**cfg["params"], **params)
    seasons = {"test": [rules.run_season(env, rule, y) for y in test_years]}
    write_seasons(out, seasons)
    return {"params": params, **summary(seasons)}


def run_oracle(job: Job, plan: Plan, out: Path, store: Store) -> dict:
    res = oracle.optimize(plan.sites[job.site], job.years, float(job.budget), job.config, job.seed)
    write_csv(out / "history.csv", res.history)
    result = asdict(res)
    result.pop("history")
    return result


def run_evaluate(job: Job, plan: Plan, out: Path, store: Store) -> dict:
    """E5: a trained model again on the test years, with noisy observations."""
    experiment, source_id = job.source.split("/")
    source = store.job_dir(experiment, source_id)
    if not (source / "model.zip").exists():
        raise FileNotFoundError(f"model of {job.source} not found; run that job first")
    _, _, test_years = year_split(job, plan)
    spec = env_spec(job, plan, test_years)
    if (source / "scale.json").exists():
        spec = replace(spec, scale=read_json(source / "scale.json"))
    seasons = {"test": drl.evaluate(drl.load(spec, source / "model.zip"), spec, test_years,
                                    seed=job.seed)}
    write_seasons(out, seasons)
    return summary(seasons)


TASKS = {"drl": run_drl, "rule": run_rule, "oracle": run_oracle, "evaluate": run_evaluate}


def write_seasons(out: Path, seasons: dict[str, list[dict]]) -> None:
    """seasons.csv for every split, decisions.csv for the test seasons."""
    write_csv(out / "seasons.csv", [
        {"split": split, "year": s["year"], "yield_t_ha": s["yield_t_ha"],
         "irrigation_mm": s["irrigation_mm"], "crop_dead": s["crop_dead"]}
        for split, items in seasons.items() for s in items])
    write_csv(out / "decisions.csv", [{"year": s["year"], **d}
                                      for s in seasons["test"] for d in s["decisions"]])


def summary(seasons: dict[str, list[dict]]) -> dict:
    return {f"{split}_mean_yield_t_ha": float(np.mean([s["yield_t_ha"] for s in items]))
            for split, items in seasons.items()}
