"""Raw job folders -> result tables per experiment in results/<experiment>/ (kept in git).

    jobs.csv              one row per finished job: what was run, with which seed and code,
                          how long and how many steps it trained, and its summary numbers
    seasons.csv           one row per evaluated season: split, year, yield, water, crop death
    learning.csv          DRL: validation yield over training steps
    convergence.csv       DRL: mean yield of the training seasons per block (stopping rule)
    decisions.csv.gz      one row per decision in a test season (compressed: the largest table)
    hyperparameters.csv   DRL: the PPO settings actually used, one row per job

results/manifests/ keeps a copy of every run manifest (code version, packages, machine). An export
rebuilds the tables of an experiment from all of its finished jobs; nothing is edited by hand.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from ditrl.paths import RESULTS
from ditrl.plan import Plan
from ditrl.store import RESULT, Store, read_csv, read_json

# Method settings copied into every row, so that tables can be grouped by design choice.
LABELS = ("observation", "constraint", "reward", "depths", "interval")
PER_JOB_FILES = {"seasons.csv": "seasons.csv", "learning.csv": "learning.csv",
                 "convergence.csv": "convergence.csv", "decisions.csv": "decisions.csv.gz"}


def describe(job: dict) -> dict:
    """The columns that identify a job in every table."""
    cfg = job["config"]
    noise = cfg.get("noise") or {}
    return {"job": job["id"], "kind": job["kind"], "method": job["method"], "site": job["site"],
            "budget_mm": job["budget"], "seed": job["seed"],
            **{label: str(cfg.get(label)) for label in LABELS},
            "trained_on_year": (job["train_years"] or [None])[0],
            "oracle_year": job["years"][0] if job["years"] and len(job["years"]) == 1 else None,
            "oracle_years": len(job["years"]) if job["years"] else None,
            "oracle_events": cfg.get("events") if job["kind"] == "oracle" else None,
            "noise_sd": next(iter(noise.values()), None)}


def _flat(result: dict) -> dict:
    """Scalars as they are; lists and mappings as JSON text."""
    return {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in result.items()}


def export_experiment(store: Store, experiment: str, out: Path) -> int:
    """Rebuild the tables of one experiment; returns the number of finished jobs."""
    folders = store.finished(experiment)
    if not folders:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    jobs, parts, settings = [], {name: [] for name in PER_JOB_FILES}, []
    for folder in folders:
        job = describe(read_json(folder / "job.json"))
        jobs.append({**job, **_flat(read_json(folder / RESULT))})
        for name in PER_JOB_FILES:
            if (folder / name).exists():
                frame = read_csv(folder / name)
                parts[name].append(frame.assign(**job)[list(job) + list(frame.columns)])
        if (folder / "hyperparameters.json").exists():
            settings.append({**job, **_flat(read_json(folder / "hyperparameters.json"))})
    pd.DataFrame(jobs).to_csv(out / "jobs.csv", index=False)
    for name, target in PER_JOB_FILES.items():
        if parts[name]:
            pd.concat(parts[name], ignore_index=True).to_csv(out / target, index=False)
    if settings:
        pd.DataFrame(settings).to_csv(out / "hyperparameters.csv", index=False)
    return len(folders)


def export(plan: Plan, store: Store, experiments: list[str], out: Path = RESULTS) -> None:
    for experiment in experiments:
        count = export_experiment(store, experiment, out / experiment)
        if count:
            print(f"{experiment}: {count} of {len(plan.expand(experiment))} jobs exported")
    if store.manifests.exists():
        (out / "manifests").mkdir(parents=True, exist_ok=True)
        for manifest in store.manifests.glob("*.json"):
            shutil.copy2(manifest, out / "manifests" / manifest.name)
