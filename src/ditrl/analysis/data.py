"""Job folders -> flat tables in <results>/tables/ (regenerable, never edited by hand).

For every table named in the plan (`analysis.tables`) three files are built from the finished
jobs of its experiments: <name>_seasons.csv (one row per evaluated season), <name>_decisions.csv
(one row per decision in a test season) and <name>_learning.csv (validation curves). In addition,
oracle_ds.csv and oracle_ss.csv hold the oracle reference values.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ditrl.paths import REPO
from ditrl.plan import Plan
from ditrl.store import RESULT, Store, read_csv, read_json

# Method settings copied into every row, so that tables can be grouped by design choice.
LABELS = ("observation", "constraint", "reward", "depths", "interval")
NO_DECISIONS = {"robustness"}   # noisy re-evaluations: yields are enough
ORACLE_MODES = {"ds": "oracle_ds", "ss": "oracle_ss"}   # mode -> experiment of the own oracle


def finished(store: Store, experiments: list[str]):
    """(job, folder) of every finished job of the experiments."""
    for experiment in experiments:
        root = store.jobs / experiment
        for folder in sorted(root.glob("*")) if root.exists() else []:
            if (folder / RESULT).exists():
                yield read_json(folder / "job.json"), folder


def describe(job: dict) -> dict:
    """The columns that identify a job in the tables."""
    cfg = job["config"]
    noise = cfg.get("noise") or {}
    return {"experiment": job["experiment"], "job": job["id"], "kind": job["kind"],
            "site": job["site"], "budget_mm": job["budget"], "seed": job["seed"],
            "method": job["method"], **{label: str(cfg.get(label)) for label in LABELS},
            "trained_on_year": (job["train_years"] or [None])[0],
            "noise_sd": next(iter(noise.values()), None)}


def table(store: Store, experiments: list[str], filename: str) -> pd.DataFrame:
    """One file of every finished job, with the job's columns in front."""
    frames = []
    for job, folder in finished(store, experiments):
        if (folder / filename).exists():
            frame = read_csv(folder / filename)
            frames.append(frame.assign(**describe(job))[list(describe(job)) + list(frame.columns)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def oracle_restarts(store: Store, experiment: str) -> pd.DataFrame:
    rows = []
    for job, folder in finished(store, [experiment]):
        r = read_json(folder / RESULT)
        single_year = job["years"][0] if len(job["years"]) == 1 else None
        rows.append({"site": job["site"], "year": single_year,
                     "budget_mm": job["budget"], "events": job["config"]["events"],
                     "seed": job["seed"], "yield_t_ha": r["objective"], "used_mm": r["used_mm"],
                     "generations": r["generations"], "stopped_by": r["stopped_by"]})
    return pd.DataFrame(rows)


def best_restarts(restarts: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Best restart per case. A case is converged if the two best restarts differ by less than
    `tolerance` t/ha and the best one stopped because it stalled, not at the generation cap."""
    def best(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("yield_t_ha", ascending=False)
        gap = g["yield_t_ha"].iloc[0] - g["yield_t_ha"].iloc[1] if len(g) > 1 else float("nan")
        return pd.Series({"yield_t_ha": g["yield_t_ha"].iloc[0], "used_mm": g["used_mm"].iloc[0],
                          "restarts": len(g), "top2_gap": gap,
                          "converged": bool(gap < tolerance
                                            and g["stopped_by"].iloc[0] == "stall")})
    return restarts.groupby(["site", "year", "budget_mm"], dropna=False).apply(best).reset_index()


def oracle(plan: Plan, store: Store, mode: str) -> pd.DataFrame:
    """EA-DS or EA-SS reference values (site, [year,] budget_mm, yield_t_ha) from the source
    named in the plan."""
    cfg = plan.analysis
    if cfg["oracle_source"] == "getoptis":
        return pd.read_csv(REPO / cfg["getoptis"][mode])
    restarts = oracle_restarts(store, ORACLE_MODES[mode])
    if restarts.empty:
        return restarts
    best = best_restarts(restarts, cfg["oracle_converged_within"])
    if not best["converged"].all():
        print(f"warning: {(~best['converged']).sum()} oracle {mode} cases are not converged")
    return best


def collect(plan: Plan, store: Store) -> None:
    store.tables.mkdir(parents=True, exist_ok=True)
    files = {"seasons": "seasons.csv", "decisions": "decisions.csv", "learning": "learning.csv"}
    for name, experiments in plan.analysis["tables"].items():
        for part, filename in files.items():
            if part == "decisions" and name in NO_DECISIONS:
                continue
            frame = table(store, experiments, filename)
            if not frame.empty:
                frame.to_csv(store.tables / f"{name}_{part}.csv", index=False)
        print(f"{name}: {len(list(finished(store, experiments)))} finished jobs")
    for mode in ORACLE_MODES:
        values = oracle(plan, store, mode)
        if not values.empty:
            values.to_csv(store.tables / f"oracle_{mode}.csv", index=False)
            print(f"oracle_{mode}: {len(values)} cases ({plan.analysis['oracle_source']})")


def read(store: Store, name: str) -> pd.DataFrame:
    """A table built by `collect`; fails with a hint if it is missing."""
    path: Path = store.tables / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the experiments behind it, "
                                "then ditrl collect")
    return read_csv(path)


def test_seasons(store: Store, name: str) -> pd.DataFrame:
    df = read(store, f"{name}_seasons")
    return df[df["split"] == "test"]
