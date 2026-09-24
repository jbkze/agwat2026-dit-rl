"""Reading the YAML configs and expanding an experiment into single jobs.

A job is one unit of work with its own folder: one DRL training, one tuned heuristic, one
oracle restart, or one evaluation of a trained model. Jobs are plain dicts, so that they can be
written to `jobs.jsonl` and sent to worker processes as they are.
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import yaml

from ditrl.sites import REPO, Site

CONFIG = REPO / "config"


def load_yaml(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_sites() -> dict:
    """{"sites": {name: Site}, "prices": {...}} from config/sites.yaml."""
    raw = load_yaml(CONFIG / "sites.yaml")
    prices = raw.pop("prices")
    return {"sites": {n: Site.from_config(n, c) for n, c in raw.items()}, "prices": prices}


def agent_config(name: str, site: str, overrides: dict | None = None) -> dict:
    """Settings of one method for one site: base entry, then site keys, then overrides."""
    agents = load_yaml(CONFIG / "agents.yaml")
    cfg = copy.deepcopy(agents[name])
    if "base" in cfg:
        cfg = {**copy.deepcopy(agents[cfg.pop("base")]), **cfg}
    site_keys = cfg.pop("sites", {}).get(site, {})
    cfg.update(site_keys)
    cfg.update(overrides or {})
    if "depths" in (overrides or {}):
        cfg.pop("max_depth", None)
    return cfg


def _slug(value) -> str:
    if isinstance(value, list):
        return f"{len(value)}x{max(value):g}" if all(isinstance(v, (int, float)) for v in value) \
            else "-".join(map(str, value))
    return str(value)


def _budgets(spec, site: Site) -> list:
    return site.budgets if spec == "all" else list(spec)


def _seeds(spec, site: str) -> list[int]:
    n = spec[site] if isinstance(spec, dict) else spec
    return list(range(n))


def _years(spec: str, site: Site) -> list[int]:
    return {"train": site.train_years, "val": site.val_years, "test": site.test_years,
            "train_val": site.train_years + site.val_years}[spec]


def expand(exp: dict) -> list[dict]:
    """All jobs of an experiment config, each with a unique, readable `id`."""
    kind = exp["kind"]
    if kind == "evaluate":
        return _evaluate_jobs(exp)
    sites = load_sites()["sites"]
    jobs = []
    for site_name, budget_spec in exp["budgets"].items():
        site = sites[site_name]
        budgets = _budgets(budget_spec, site)
        if kind == "oracle":
            jobs += _oracle_jobs(exp, site, budgets)
            continue
        factor_names = list(exp.get("factors", {}))
        cells = list(itertools.product(*exp.get("factors", {}).values())) or [()]
        for agent, cell in itertools.product(exp["agents"], cells):
            overrides = dict(zip(factor_names, cell))
            cfg = agent_config(agent, site_name, {**exp.get("overrides", {}), **overrides})
            label = agent + "".join(f"_{k}-{_slug(v)}" for k, v in overrides.items())
            is_rule = "rule" in cfg
            job_budgets = [None] if is_rule and cfg.get("budget", 0) is None else budgets
            train_years = [[y] for y in site.test_years] if exp.get("train_on_test_year") \
                else [None]
            seeds = [0] if is_rule else _seeds(exp.get("seeds", 10), site_name)
            for budget, years, seed in itertools.product(job_budgets, train_years, seeds):
                yid = f"_y{years[0]}" if years else ""
                bid = "unlimited" if budget is None else f"B{budget:g}"
                jobs.append({"id": f"{site_name}_{bid}_{label}{yid}_s{seed}",
                             "kind": "heuristic" if is_rule else "drl", "site": site_name,
                             "budget": budget, "agent": agent, "overrides": overrides,
                             "config": cfg, "seed": seed, "train_years": years})
    return jobs


def _oracle_jobs(exp: dict, site: Site, budgets: list) -> list[dict]:
    """EA-DS: one case per (year, budget). EA-SS: one case per budget over all `years`."""
    years = _years(exp["years"], site)
    cases = [[y] for y in years] if exp["mode"] == "ds" else [years]
    restarts = itertools.product(exp["restarts"]["n_events"], exp["restarts"]["seeds"])
    jobs = []
    for (n_events, seed), case, budget in itertools.product(restarts, cases, budgets):
        cid = f"y{case[0]}" if exp["mode"] == "ds" else f"{exp['years']}{len(case)}"
        jobs.append({"id": f"{site.name}_B{budget:g}_{exp['mode']}_{cid}_e{n_events}_s{seed}",
                     "kind": "oracle", "site": site.name, "budget": budget, "years": case,
                     "mode": exp["mode"], "n_events": n_events, "seed": seed,
                     "window": exp["window_days"][site.name], "ga": exp["ga"],
                     "quantile": exp.get("quantile")})
    return jobs


def _evaluate_jobs(exp: dict) -> list[dict]:
    """E5: evaluate every finished DRL job of the source runs again, with observation noise."""
    jobs = []
    for run in exp["source_runs"]:
        run_dir = REPO / run
        for line in (run_dir / "jobs.jsonl").read_text().splitlines():
            src = json.loads(line)
            if src["kind"] != "drl" or not (run_dir / "jobs" / src["id"] / "result.json").exists():
                continue
            for sd, noise_seed in itertools.product(exp["noise"]["sd"],
                                                    range(exp["noise"]["seeds"])):
                jobs.append({"id": f"{src['id']}_noise{sd:g}_n{noise_seed}", "kind": "evaluate",
                             "source": str(Path(run) / "jobs" / src["id"]), "source_job": src,
                             "noise": {f: sd for f in exp["noise"]["features"]},
                             "seed": noise_seed})
    return jobs
