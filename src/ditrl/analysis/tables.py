"""LaTeX table bodies for the paper. Caption and label stay in the paper; each generator writes only
the tabular environment (booktabs), plus a CSV with the same numbers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ditrl.analysis import style
from ditrl.analysis.collect import read, test_seasons
from ditrl.analysis.stats import main_effects, paired_differences, per_policy
from ditrl.plan import Plan
from ditrl.store import Store, read_json

DECIMALS = "%.2f"
PPO_SETTINGS = ["algorithm", "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
                "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "net_arch"]


def _write(df: pd.DataFrame, out: Path, colspec: str) -> None:
    df.to_csv(out.with_suffix(".csv"), index=False)
    body = df.to_latex(index=False, escape=True, column_format=colspec, float_format=DECIMALS)
    out.write_text(body.strip() + "\n", encoding="utf-8")


def ablation(plan: Plan, store: Store, out: Path) -> None:
    """E2, per variant and case: mean, lowest and highest paired difference to the reference over
    seeds, for the mean test yield and (with several test years) its lower quantile."""
    cfg = plan.analysis
    diffs = paired_differences(test_seasons(store, "ablation"), cfg["ablation_reference"],
                               cfg["lower_quantile"])
    t = (diffs.groupby(["method", "site", "budget_mm"])
         .agg(seeds=("seed", "size"), mean=("mean_difference", "mean"),
              lowest=("mean_difference", "min"), highest=("mean_difference", "max"),
              low_quantile=("low_difference", "mean"), multi_year=("years", "max"))
         .reset_index())
    t.loc[t["multi_year"] == 1, "low_quantile"] = np.nan   # one test year: no quantile
    t = t.drop(columns="multi_year")
    t.columns = ["Variant", "Site", "Budget (mm)", "Seeds", "Mean (t/ha)", "Lowest", "Highest",
                 f"{cfg['lower_quantile']:.0%} quantile (t/ha)"]
    _write(t, out, "llrrrrrr")


def ablation_effects(plan: Plan, store: Store, out: Path) -> None:
    """E2 from step C on: main effect of each design choice, paired over everything else."""
    cfg = plan.analysis
    seasons = test_seasons(store, "ablation")
    factors = {f: list(levels) for f, levels in cfg["ablation_factors"].items()}
    reference = {f: str(plan.method(cfg["ablation_reference"], site)[f])
                 for f in factors for site in plan.sites}
    t = main_effects(seasons, factors, reference, cfg["bootstrap"], cfg["wilcoxon_min_pairs"])
    names = cfg["ablation_factors"]
    t["level"] = [names[f][lvl] for f, lvl in zip(t["factor"], t["level"], strict=True)]
    t["interval"] = [f"[{lo:.2f}, {hi:.2f}]"
                     for lo, hi in zip(t["ci_low"], t["ci_high"], strict=True)]
    t = t[["factor", "level", "site", "budget_mm", "effect_t_ha", "interval", "p_holm"]]
    t.columns = ["Factor", "Changed to", "Site", "Budget (mm)", "Effect (t/ha)", "95 % interval",
                 "p (Holm)"]
    _write(t, out, "lllrrrr")


def information(plan: Plan, store: Store, out: Path) -> None:
    """E3: mean test yield from no weather knowledge to full knowledge, per site and budget."""
    def mean_yield(df: pd.DataFrame, name: str) -> pd.Series:
        return df.groupby(["site", "budget_mm"])["yield_t_ha"].mean().rename(name)
    benchmark = test_seasons(store, "benchmark")
    cols = [mean_yield(benchmark[benchmark["method"] == m], style.label(m))
            for m in ("threshold", "fixed_interval", "drl1")]
    cols.append(mean_yield(test_seasons(store, "forecast"), style.label("drl1_forecast")))
    cols.append(mean_yield(test_seasons(store, "test_year"), "DRL1 trained on the test year"))
    cols.append(read(store, "oracle_ds").groupby(["site", "budget_mm"])["yield_t_ha"].mean()
                .rename("EA-DS"))
    if (store.tables / "oracle_ss.csv").exists():
        cols.insert(0, read(store, "oracle_ss").groupby(["site", "budget_mm"])["yield_t_ha"]
                    .mean().rename("EA-SS"))
    t = pd.concat(cols, axis=1).reset_index()
    t.columns = ["Site", "Budget (mm)", *t.columns[2:]]
    _write(t, out, "lr" + "r" * (len(t.columns) - 2))


def robustness(plan: Plan, store: Store, out: Path) -> None:
    """E5, Nebraska: mean and lower quantile of test yield per method (median over seeds), and
    the mean yield under observation noise."""
    q = plan.analysis["lower_quantile"]
    benchmark = test_seasons(store, "benchmark")
    t = per_policy(benchmark[benchmark["site"] == "nebraska"], q)
    t = t.groupby(["budget_mm", "method"])[["mean_yield", "low_yield"]].median().reset_index()
    noisy = test_seasons(store, "robustness")
    for sd, g in noisy[noisy["site"] == "nebraska"].groupby("noise_sd"):
        noise = g.groupby(["budget_mm", "method"])["yield_t_ha"].mean().rename(f"noise {sd:.0%}")
        t = t.merge(noise.reset_index(), on=["budget_mm", "method"], how="left")
    t["method"] = t["method"].map(style.label)
    t = t.rename(columns={"budget_mm": "Budget (mm)", "method": "Method", "mean_yield": "Mean",
                          "low_yield": f"{q:.0%} quantile"})
    _write(t, out, "rl" + "r" * (len(t.columns) - 2))


def hyperparameters(plan: Plan, store: Store, out: Path) -> None:
    """The PPO settings actually used, read from the job folders of the benchmark."""
    cols = {}
    for method in ("drl1", "drl2"):
        for site in plan.sites:
            pattern = f"{site}_*_{method}_s*/hyperparameters.json"
            files = sorted((store.jobs / "benchmark").glob(pattern))
            if files:
                cols[f"{style.label(method)} {site.capitalize()}"] = read_json(files[0])
    t = pd.DataFrame({name: [p.get(k) for k in PPO_SETTINGS] for name, p in cols.items()},
                     index=PPO_SETTINGS)
    t = t.map(lambda v: f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v))
    t.index.name = "Parameter"
    _write(t.reset_index(), out, "l" * (len(cols) + 1))
