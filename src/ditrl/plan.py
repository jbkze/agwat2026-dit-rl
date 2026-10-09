"""The experiment plan (config/plan.yaml) and the jobs it expands into.

A job is one unit of work with its own result folder: one DRL training, one tuned rule, one oracle
restart, or one evaluation of a trained model. Expansion depends only on the plan, never on
results, so every machine derives the same job list and the same shards from it.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ditrl.paths import PLAN
from ditrl.sites import Site, inclusive_range

STAGE_PREFIX = "stage"


@dataclass
class Job:
    experiment: str
    id: str
    kind: str                         # drl | rule | oracle | evaluate
    site: str
    seed: int
    budget: float | None = None       # None: unlimited
    method: str | None = None
    config: dict = field(default_factory=dict)   # method settings, oracle settings or noise
    train_years: list[int] | None = None         # train and test on these years only (E3)
    years: list[int] | None = None               # oracle: years of the case
    source: str | None = None                    # evaluate: "<experiment>/<job id>" of the model

    def to_dict(self) -> dict:
        return asdict(self)


class Plan:
    def __init__(self, raw: dict):
        self.raw = raw
        self.sites = {name: Site.from_config(name, cfg) for name, cfg in raw["sites"].items()}
        self.prices = raw["prices"]
        self.methods = raw["methods"]
        self.oracle = raw["oracle"]
        self.experiments = raw["experiments"]
        self.analysis = raw["analysis"]

    @classmethod
    def load(cls, path: Path = PLAN) -> Plan:
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")))

    # --- methods ----------------------------------------------------------------------------------
    def method(self, name: str, site: str, overrides: dict | None = None) -> dict:
        """Settings of a method at a site: its base chain, then its site keys, then overrides."""
        entry = copy.deepcopy(self.methods[name])
        base = entry.pop("base", None)
        cfg = self.method(base, site) if base else {}
        site_keys = entry.pop("sites", {}).get(site, {})
        return {**cfg, **entry, **site_keys, **(overrides or {})}

    # --- selection --------------------------------------------------------------------------------
    def select(self, names: list[str]) -> list[str]:
        """Experiment names for a selection of stages ("stage1") and experiment names."""
        chosen = []
        for name in names:
            if name.startswith(STAGE_PREFIX):
                stage = int(name.removeprefix(STAGE_PREFIX))
                found = [e for e, cfg in self.experiments.items() if cfg.get("stage") == stage]
            elif name in self.experiments:
                found = [name]
            else:
                raise KeyError(f"{name!r} is neither a stage nor an experiment in the plan")
            chosen += [e for e in found if e not in chosen]
        return chosen

    def jobs(self, names: list[str]) -> list[Job]:
        return [job for experiment in self.select(names) for job in self.expand(experiment)]

    # --- expansion --------------------------------------------------------------------------------
    def expand(self, experiment: str) -> list[Job]:
        exp = self.experiments[experiment]
        if exp["kind"] == "evaluate":
            return self._evaluate_jobs(experiment, exp)
        jobs = []
        for site_name, budget_spec in exp["budgets"].items():
            site = self.sites[site_name]
            budgets = site.budgets if budget_spec == "all" else budget_spec
            seeds = _seeds(exp["seeds"], site_name)
            if exp["kind"] == "oracle":
                jobs += self._oracle_jobs(experiment, exp, site, budgets, seeds)
            else:
                jobs += self._method_jobs(experiment, exp, site, budgets, seeds)
        return jobs

    def _method_jobs(self, experiment: str, exp: dict, site: Site, budgets: list,
                     seeds: list[int]) -> list[Job]:
        jobs = []
        year_sets = [[y] for y in site.test_years] if exp.get("train_on_test_year") else [None]
        for method in exp["methods"]:
            cfg = self.method(method, site.name, exp.get("overrides"))
            is_rule = "rule" in cfg
            job_budgets = [None] if cfg.get("unlimited") else budgets
            job_seeds = seeds[:1] if is_rule else seeds   # rules are deterministic
            for budget, years, seed in itertools.product(job_budgets, year_sets, job_seeds):
                year_tag = f"_y{years[0]}" if years else ""
                jobs.append(Job(experiment=experiment,
                                id=f"{site.name}_{_budget_tag(budget)}_{method}{year_tag}_s{seed}",
                                kind="rule" if is_rule else "drl", site=site.name, seed=seed,
                                budget=budget, method=method, config=cfg, train_years=years))
        return jobs

    def _oracle_jobs(self, experiment: str, exp: dict, site: Site, budgets: list,
                     seeds: list[int]) -> list[Job]:
        years = site.years(exp["years"])
        cases = [[y] for y in years] if exp["per_year"] else [years]
        cfg = {**self.oracle, **exp.get("overrides", {}), "quantile": exp.get("quantile"),
               "window_days": self.oracle["window_days"][site.name]}
        jobs = []
        for events, seed, case, budget in itertools.product(exp["events"], seeds, cases, budgets):
            case_tag = f"y{case[0]}" if exp["per_year"] else f"{exp['years']}{len(case)}"
            jobs.append(Job(experiment=experiment,
                            id=f"{site.name}_{_budget_tag(budget)}_{case_tag}_e{events}_s{seed}",
                            kind="oracle", site=site.name, seed=seed, budget=budget,
                            config={**cfg, "events": events}, years=case))
        return jobs

    def _evaluate_jobs(self, experiment: str, exp: dict) -> list[Job]:
        """Every DRL job of the source experiments again on the test years, with noisy
        observations of the given features. Each one needs that job's trained model."""
        jobs = []
        for source in self.jobs(exp["source"]):
            if source.kind != "drl":
                continue
            for sd, seed in itertools.product(exp["noise_sd"], inclusive_range(exp["seeds"])):
                noise = {feature: sd for feature in exp["noise_features"]}
                jobs.append(Job(experiment=experiment, id=f"{source.id}_noise{sd:g}_n{seed}",
                                kind="evaluate", site=source.site, seed=seed,
                                budget=source.budget, method=source.method,
                                config={**source.config, "noise": noise},
                                source=f"{source.experiment}/{source.id}"))
        return jobs


def shard(jobs: list[Job], index: int, count: int) -> list[Job]:
    """Shard `index` of `count` (1-based): every count-th job, so that each shard gets a similar
    mix of sites and methods."""
    if not 1 <= index <= count:
        raise ValueError(f"shard {index}/{count} does not exist")
    return jobs[index - 1 :: count]


def _seeds(spec: list[int] | dict, site: str) -> list[int]:
    """An inclusive seed range, or one range per site."""
    return inclusive_range(spec[site] if isinstance(spec, dict) else spec)


def _budget_tag(budget: float | None) -> str:
    return "unlimited" if budget is None else f"B{budget:g}"
