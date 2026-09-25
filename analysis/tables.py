"""LaTeX table bodies for the paper. Caption and label stay in the paper; each generator writes
only the tabular environment (booktabs), plus a CSV with the same numbers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common import METHOD_LABEL, derived, run_dir, test_seasons
from analysis.fig_ablation import LABELS, effects
from analysis.stats import lower_tail


def _write(df: pd.DataFrame, out: Path, colspec: str) -> None:
    df.to_csv(out.with_suffix(".csv"), index=False)
    body = df.to_latex(index=False, escape=True, column_format=colspec, float_format="%.2f")
    out.write_text(body.strip() + "\n", encoding="utf-8")


def ablation(out: Path) -> None:
    """E2: yield change per factor, site and budget with 95 % interval and Holm-adjusted p."""
    t = effects()
    t["factor"] = t["factor"].map(LABELS)
    t["interval"] = t.apply(lambda r: f"[{r.ci_low:.2f}, {r.ci_high:.2f}]", axis=1)
    t = t[["factor", "site", "budget_mm", "effect_t_ha", "interval", "p_holm"]]
    t.columns = ["Change", "Site", "Budget (mm)", "Effect (t/ha)", "95 % interval", "p (Holm)"]
    _write(t, out, "llrrrr")


def information(out: Path) -> None:
    """E3: mean test yield from no weather knowledge to full knowledge, per site and budget."""
    def mean_yield(df, label):
        return df.groupby(["site", "budget_mm"])["yield_t_ha"].mean().rename(label)
    e1 = test_seasons("benchmark")
    cols = [mean_yield(e1[e1["agent"] == a], METHOD_LABEL[a])
            for a in ("threshold", "fixed_interval", "drl1")]
    cols.append(mean_yield(test_seasons("forecast"), "DRL1 + 7-day forecast"))
    cols.append(mean_yield(test_seasons("test_year"), "DRL1 trained on test year"))
    cols.append(derived("oracle_ds").groupby(["site", "budget_mm"])["yield_t_ha"].mean()
                .rename("EA-DS"))
    try:
        cols.insert(0, derived("oracle_ss").groupby(["site", "budget_mm"])["yield_t_ha"].mean()
                    .rename("EA-SS"))
    except FileNotFoundError:
        pass
    t = pd.concat(cols, axis=1).reset_index()
    t.columns = ["Site", "Budget (mm)", *t.columns[2:]]
    _write(t, out, "lr" + "r" * (len(t.columns) - 2))


def robustness(out: Path) -> None:
    """E5: mean, 10 % quantile and CVaR of test yield per method (median over seeds), and the
    mean yield under observation noise."""
    e1 = test_seasons("benchmark")
    rows = []
    for (site, budget, agent, seed), g in e1.groupby(["site", "budget_mm", "agent", "seed"]):
        if site == "nebraska":
            rows.append({"site": site, "budget_mm": budget, "agent": agent,
                         "mean": g["yield_t_ha"].mean(), **lower_tail(g["yield_t_ha"])})
    t = pd.DataFrame(rows).groupby(["site", "budget_mm", "agent"]).median().reset_index()
    noisy = test_seasons("robustness")
    noisy = noisy[noisy["site"] == "nebraska"]
    for sd, g in noisy.groupby("noise_sd"):
        m = g.groupby(["budget_mm", "agent"])["yield_t_ha"].mean().rename(f"noise {sd:.0%}")
        t = t.merge(m.reset_index(), on=["budget_mm", "agent"], how="left")
    t["agent"] = t["agent"].map(lambda a: METHOD_LABEL.get(a, a))
    t = t.drop(columns="site").rename(columns={"budget_mm": "Budget (mm)", "agent": "Method",
                                               "mean": "Mean", "q10": "10 % quantile",
                                               "cvar10": "CVaR 10 %"})
    _write(t, out, "rl" + "r" * (len(t.columns) - 2))


def hyperparameters(out: Path) -> None:
    """R2#3: the PPO settings actually used, read from the run folders of E1."""
    jobs_dir = run_dir("benchmark") / "jobs"
    cols = {}
    for agent in ("drl1", "drl2"):
        for site in ("tunis", "nebraska"):
            files = sorted(jobs_dir.glob(f"{site}_*_{agent}_s0/hyperparameters.json"))
            if files:
                cols[f"{METHOD_LABEL[agent]} {site.capitalize()}"] = json.loads(files[0].read_text())
    keys = ["algorithm", "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
            "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "net_arch"]
    t = pd.DataFrame({name: [p.get(k) for k in keys] for name, p in cols.items()}, index=keys)
    t = t.map(lambda v: f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v))
    t.index.name = "Parameter"
    _write(t.reset_index(), out, "l" + "l" * len(cols))
