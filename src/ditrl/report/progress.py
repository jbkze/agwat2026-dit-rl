"""Progress: how many jobs of each experiment of the plan have results."""

from __future__ import annotations

import pandas as pd

from ditrl.report.core import Context, Entry, Section, write_table


def counts(ctx: Context) -> pd.DataFrame:
    """Planned and finished jobs per experiment that belongs to a stage."""
    rows = []
    for name, cfg in ctx.plan.experiments.items():
        if cfg.get("stage") is None:
            continue
        jobs = ctx.plan.expand(name)
        done = ctx.table(name, "jobs")
        rows.append({"Stage": cfg["stage"], "Experiment": name, "Kind": cfg["kind"],
                     "Jobs": len(jobs), "Done": 0 if done is None else len(done)})
    table = pd.DataFrame(rows).sort_values(["Stage"], kind="stable")
    table["Share"] = table["Done"] / table["Jobs"]
    return table


def build(ctx: Context) -> Section:
    table = counts(ctx)
    shown = table.assign(Share=(100 * table["Share"]).round().astype(int).astype(str) + " %")
    path = ctx.tables / "progress.md"
    write_table(shown, path, decimals=0)
    trainings = table[table["Kind"] == "methods"]
    finding = (f"{int(trainings['Done'].sum())} of {int(trainings['Jobs'].sum())} trainings and "
               f"rule runs have results.")
    return Section(entries=[Entry(path, "Progress", "Planned and finished jobs per experiment.",
                                  __name__)], findings=[finding])
