"""The result tables of results/<experiment>/, combined as the analysis needs them, and readable
names for the methods."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ditrl.analysis import style
from ditrl.paths import REPO
from ditrl.plan import Plan
from ditrl.store import read_csv

ORACLE_EXPERIMENTS = {"ds": "oracle_ds", "ss": "oracle_ss"}


def _part(results: Path, experiment: str, part: str) -> pd.DataFrame | None:
    for path in (results / experiment / f"{part}.csv", results / experiment / f"{part}.csv.gz"):
        if path.exists():
            return read_csv(path).assign(experiment=experiment)
    return None


def table(plan: Plan, results: Path, name: str, part: str) -> pd.DataFrame:
    """One part (jobs, seasons, learning, decisions, hyperparameters) of every experiment that
    feeds the analysis table `name` in the plan; fails with a hint if none has results yet."""
    found = (_part(results, experiment, part) for experiment in plan.analysis["tables"][name])
    frames = [frame for frame in found if frame is not None]
    if not frames:
        raise FileNotFoundError(f"no {part} for {name!r} in {results}: "
                                "run and export its experiments")
    return pd.concat(frames, ignore_index=True)


def test_seasons(plan: Plan, results: Path, name: str) -> pd.DataFrame:
    seasons = table(plan, results, name, "seasons")
    return seasons[seasons["split"] == "test"]


def best_restarts(restarts: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Best oracle restart per case. A case is converged if the two best restarts differ by less
    than `tolerance` t/ha and the best one stopped because it stalled, not at the generation cap."""
    def best(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("objective", ascending=False)
        gap = g["objective"].iloc[0] - g["objective"].iloc[1] if len(g) > 1 else float("nan")
        return pd.Series({"yield_t_ha": g["objective"].iloc[0], "used_mm": g["used_mm"].iloc[0],
                          "restarts": len(g), "top2_gap": gap,
                          "converged": bool(gap < tolerance
                                            and g["stopped_by"].iloc[0] == "stall")})
    cases = restarts.rename(columns={"oracle_year": "year"})
    return cases.groupby(["site", "year", "budget_mm"], dropna=False).apply(best).reset_index()


def oracle(plan: Plan, results: Path, mode: str) -> pd.DataFrame:
    """EA-DS ("ds") or EA-SS ("ss") reference values: site, [year,] budget_mm, yield_t_ha, from
    the source named in the plan."""
    cfg = plan.analysis
    if cfg["oracle_source"] == "getoptis":
        return pd.read_csv(REPO / cfg["getoptis"][mode])
    restarts = _part(results, ORACLE_EXPERIMENTS[mode], "jobs")
    if restarts is None:
        raise FileNotFoundError(f"no results of {ORACLE_EXPERIMENTS[mode]} yet")
    best = best_restarts(restarts, cfg["oracle_converged_within"])
    if not best["converged"].all():
        print(f"warning: {(~best['converged']).sum()} oracle {mode} cases are not converged")
    return best


def method_label(plan: Plan, method: str) -> str:
    """A readable name: the fixed name of a main method, or DRL1 plus what the variant changes."""
    if method in style.METHODS:
        return style.METHODS[method][0]
    reference = plan.analysis["ablation_reference"]
    site = next(iter(plan.sites))
    cfg, base = plan.method(method, site), plan.method(reference, site)
    changes = [names[str(cfg[f])] for f, names in plan.analysis["ablation_factors"].items()
               if str(cfg[f]) != str(base[f])]
    if cfg["depths"] != base["depths"]:
        changes.append(f"doses up to {max(cfg['depths']):g} mm")
    if cfg["interval"] != base["interval"]:
        changes.append("daily decisions" if cfg["interval"] == 1
                       else f"decisions every {cfg['interval']} days")
    if cfg.get("normalize") != base.get("normalize"):
        changes.append("scaled observations")
    return ", ".join([style.label(reference), *changes])
