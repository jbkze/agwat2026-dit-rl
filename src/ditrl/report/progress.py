"""Status: one row per experiment of the plan, what it tests, how many of its runs are done, and
where its results are. Experiments without results stay in the table as pending."""

from __future__ import annotations

import pandas as pd

from ditrl.report.core import Context, Entry, Section, write_table


def counts(ctx: Context, link_base: str = "") -> pd.DataFrame:
    rows = []
    for name, cfg in ctx.plan.experiments.items():
        if cfg.get("stage") is None:
            continue
        planned = len(ctx.plan.expand(name))
        done = ctx.table(name, "jobs")
        done = 0 if done is None else len(done)
        rows.append({"Stage": cfg["stage"], "Experiment": f"`{name}`",
                     "What it tests": cfg.get("about", ""), "Runs": f"{done} of {planned}",
                     "Results": f"[results/{name}/]({link_base}results/{name}/)" if done
                     else "pending"})
    return pd.DataFrame(rows).sort_values("Stage", kind="stable")


def build(ctx: Context) -> Section:
    path = ctx.tables / "progress.md"
    write_table(counts(ctx, "../../"), path, decimals=0)
    ctx.facts["status"] = counts(ctx)
    return Section("Status", [Entry(path, "Status",
                                    "Every experiment of the plan with its finished runs.",
                                    __name__)])
