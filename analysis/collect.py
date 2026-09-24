"""Turn run folders into flat tables in results/derived/ (regenerable, never edited by hand).

    python -m analysis.collect          # rebuild all derived tables named in config/analysis.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ditrl.config import latest_run

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "results" / "derived"
FACTORS = ("observation", "constraint", "reward", "depths", "interval")


def analysis_config() -> dict:
    """config/analysis.yaml with every `latest` replaced by the newest matching run folder
    (or None if that experiment has not run yet)."""
    cfg = yaml.safe_load((REPO / "config" / "analysis.yaml").read_text(encoding="utf-8"))

    def resolve(value, experiment):
        if value != "latest":
            return value
        run = latest_run(experiment)
        return str(run.relative_to(REPO)) if run else None

    cfg["runs"] = {k: resolve(v, k) for k, v in cfg["runs"].items()}
    pymoo = cfg["oracle"]["pymoo"]
    cfg["oracle"]["pymoo"] = {m: resolve(v, f"e0_oracle_{m}") for m, v in pymoo.items()}
    return cfg


def _jobs(run_dir: Path):
    for line in (run_dir / "jobs.jsonl").read_text().splitlines():
        job = json.loads(line)
        folder = run_dir / "jobs" / job["id"]
        if (folder / "result.json").exists():
            yield job, folder


def _label(job: dict) -> dict:
    cfg = job["config"]
    return {"agent": job["agent"], **{f: str(cfg.get(f)) for f in FACTORS},
            "trained_on_year": (job.get("train_years") or [None])[0]}


def seasons(run: str | Path) -> pd.DataFrame:
    """One row per (job, split, year): yield, water use and crop death."""
    run_dir = REPO / run
    rows = []
    for job, folder in _jobs(run_dir):
        seasons_ = json.loads((folder / "seasons.json").read_text())
        base = {"run": run_dir.name, "job": job["id"], "kind": job["kind"], "site": job["site"],
                "budget_mm": job.get("budget"), "seed": job["seed"]}
        base.update(_label(job.get("source_job", job)))
        if job["kind"] == "evaluate":
            base["noise_sd"] = next(iter(job["noise"].values()))
        for split, items in seasons_.items():
            for s in items:
                rows.append({**base, "split": split, "year": s["year"],
                             "yield_t_ha": s["yield_t_ha"], "irrigation_mm": s["irrigation_mm"],
                             "crop_dead": s["crop_dead"]})
    return pd.DataFrame(rows)


def decisions(run: str | Path) -> pd.DataFrame:
    """One row per decision in the test seasons: proposed and applied depth, remaining budget."""
    run_dir = REPO / run
    rows = []
    for job, folder in _jobs(run_dir):
        base = {"job": job["id"], "site": job["site"], "budget_mm": job.get("budget"),
                "seed": job["seed"], **_label(job)}
        for s in json.loads((folder / "seasons.json").read_text())["test"]:
            rows += [{**base, "year": s["year"], **d} for d in s["decisions"]]
    return pd.DataFrame(rows)


def learning_curves(run: str | Path) -> pd.DataFrame:
    run_dir = REPO / run
    rows = []
    for job, folder in _jobs(run_dir):
        curve = folder / "validation_curve.json"
        if curve.exists():
            base = {"job": job["id"], "site": job["site"], "budget_mm": job["budget"],
                    "seed": job["seed"], **_label(job)}
            rows += [{**base, **point} for point in json.loads(curve.read_text())]
    return pd.DataFrame(rows)


def oracle_restarts(run: str | Path) -> pd.DataFrame:
    run_dir = REPO / run
    rows = []
    for job, folder in _jobs(run_dir):
        r = json.loads((folder / "result.json").read_text())
        rows.append({"site": job["site"], "mode": job["mode"],
                     "year": job["years"][0] if job["mode"] == "ds" else None,
                     "budget_mm": job["budget"], "n_events": job["n_events"], "seed": job["seed"],
                     "yield_t_ha": r["objective"], "used_mm": r["used_mm"],
                     "generations": r["generations"], "stopped_by": r["stopped_by"]})
    return pd.DataFrame(rows)


def best_of_restarts(restarts: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Best restart per case. A case is converged if the two best restarts differ by less than
    `tolerance` t/ha and the best one stopped because it stalled (not at the generation cap)."""
    def summarize(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("yield_t_ha", ascending=False)
        top2_gap = g["yield_t_ha"].iloc[0] - g["yield_t_ha"].iloc[1] if len(g) > 1 else float("nan")
        return pd.Series({"yield_t_ha": g["yield_t_ha"].iloc[0], "used_mm": g["used_mm"].iloc[0],
                          "restarts": len(g), "top2_gap": top2_gap,
                          "converged": bool(top2_gap < tolerance
                                            and g["stopped_by"].iloc[0] == "stall")})
    keys = ["site", "mode", "year", "budget_mm"]
    return restarts.groupby(keys, dropna=False).apply(summarize).reset_index()


def oracle(mode: str = "ds") -> pd.DataFrame:
    """EA-DS or EA-SS reference values from the source named in config/analysis.yaml.
    Columns: site, year (EA-DS only), budget_mm, yield_t_ha."""
    cfg = analysis_config()["oracle"]
    if cfg["source"] == "getoptis":
        return pd.read_csv(REPO / cfg["getoptis"][mode])
    table = best_of_restarts(oracle_restarts(cfg["pymoo"][mode]), cfg["converged_if_top2_within"])
    if not table["converged"].all():
        print(f"warning: {(~table['converged']).sum()} oracle cases are not converged")
    return table


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    cfg = analysis_config()
    for name, run in cfg["runs"].items():
        if run:
            seasons(run).to_csv(DERIVED / f"{name}_seasons.csv", index=False)
            if name != "e5_robustness":  # noisy re-evaluations: yields are enough
                decisions(run).to_csv(DERIVED / f"{name}_decisions.csv", index=False)
            learning_curves(run).to_csv(DERIVED / f"{name}_learning.csv", index=False)
            print(f"collected {name} from {run}")
    for mode in ("ds", "ss"):
        if cfg["oracle"]["source"] == "getoptis" or cfg["oracle"]["pymoo"][mode]:
            oracle(mode).to_csv(DERIVED / f"oracle_{mode}.csv", index=False)
            print(f"collected oracle {mode} ({cfg['oracle']['source']})")


if __name__ == "__main__":
    main()
