"""LaTeX table bodies for the paper. Caption and label stay in the paper; each generator writes only
the tabular environment (booktabs), plus a CSV with the same numbers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ditrl.analysis.data import method_label, oracle, table, test_seasons
from ditrl.analysis.stats import main_effects, paired_differences, per_policy
from ditrl.plan import Plan

DECIMALS = "%.2f"
PPO_SETTINGS = ["algorithm", "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
                "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "net_arch"]


def _write(df: pd.DataFrame, out: Path, colspec: str) -> None:
    if "Site" in df:
        df = df.assign(Site=df["Site"].str.capitalize())
    df.to_csv(out.with_suffix(".csv"), index=False)
    body = df.to_latex(index=False, escape=True, column_format=colspec, float_format=DECIMALS)
    out.write_text(body.strip() + "\n", encoding="utf-8")


def ablation(plan: Plan, results: Path, out: Path) -> None:
    """E2, per variant and case: mean, lowest and highest paired difference to the reference over
    seeds, for the mean test yield and (with several test years) its lower quantile."""
    cfg = plan.analysis
    diffs = paired_differences(test_seasons(plan, results, "ablation"), cfg["ablation_reference"],
                               cfg["lower_quantile"])
    t = (diffs.groupby(["method", "site", "budget_mm"])
         .agg(seeds=("seed", "size"), mean=("mean_difference", "mean"),
              lowest=("mean_difference", "min"), highest=("mean_difference", "max"),
              low_quantile=("low_difference", "mean"), multi_year=("years", "max"))
         .reset_index())
    t.loc[t["multi_year"] == 1, "low_quantile"] = np.nan   # one test year: no quantile
    t = t.drop(columns="multi_year")
    t["method"] = t["method"].map(lambda m: method_label(plan, m))
    t.columns = ["Variant", "Site", "Budget (mm)", "Seeds", "Mean (t/ha)", "Lowest", "Highest",
                 f"{cfg['lower_quantile']:.0%} quantile (t/ha)"]
    _write(t, out, "llrrrrrr")


def ablation_effects(plan: Plan, results: Path, out: Path) -> None:
    """E2, once the interactions have run: main effect of each design choice, paired over
    everything else."""
    cfg = plan.analysis
    factors = {f: list(levels) for f, levels in cfg["ablation_factors"].items()}
    site = next(iter(plan.sites))
    reference = {f: str(plan.method(cfg["ablation_reference"], site)[f]) for f in factors}
    t = main_effects(test_seasons(plan, results, "ablation"), factors, reference,
                     cfg["bootstrap"], cfg["wilcoxon_min_pairs"])
    names = cfg["ablation_factors"]
    t["level"] = [names[f][lvl] for f, lvl in zip(t["factor"], t["level"], strict=True)]
    t["interval"] = [f"[{lo:.2f}, {hi:.2f}]"
                     for lo, hi in zip(t["ci_low"], t["ci_high"], strict=True)]
    t = t[["factor", "level", "site", "budget_mm", "effect_t_ha", "interval", "p_holm"]]
    t.columns = ["Factor", "Changed to", "Site", "Budget (mm)", "Effect (t/ha)", "95 % interval",
                 "p (Holm)"]
    _write(t, out, "lllrrrr")


def information(plan: Plan, results: Path, out: Path) -> None:
    """E3: mean test yield from no weather knowledge to full knowledge, per site and budget."""
    def mean_yield(df: pd.DataFrame, name: str) -> pd.Series:
        return df.groupby(["site", "budget_mm"])["yield_t_ha"].mean().rename(name)
    benchmark = test_seasons(plan, results, "benchmark")
    cols = [mean_yield(benchmark[benchmark["method"] == m], method_label(plan, m))
            for m in ("threshold", "fixed_interval", "drl1")]
    cols.append(mean_yield(test_seasons(plan, results, "forecast"),
                           method_label(plan, "drl1_forecast")))
    cols.append(mean_yield(test_seasons(plan, results, "test_year"),
                           "DRL1 trained on the test year"))
    cols.append(mean_yield(oracle(plan, results, "ds"), "EA-DS"))
    try:
        cols.insert(0, mean_yield(oracle(plan, results, "ss"), "EA-SS"))
    except FileNotFoundError:
        pass
    t = pd.concat(cols, axis=1).reset_index()
    t.columns = ["Site", "Budget (mm)", *t.columns[2:]]
    _write(t, out, "lr" + "r" * (len(t.columns) - 2))


def robustness(plan: Plan, results: Path, out: Path) -> None:
    """E5, Nebraska: mean and lower quantile of test yield per method (median over seeds), and
    the mean yield under observation noise."""
    q = plan.analysis["lower_quantile"]
    benchmark = test_seasons(plan, results, "benchmark")
    t = per_policy(benchmark[benchmark["site"] == "nebraska"], q)
    t = t.groupby(["budget_mm", "method"])[["mean_yield", "low_yield"]].median().reset_index()
    noisy = test_seasons(plan, results, "robustness")
    for sd, g in noisy[noisy["site"] == "nebraska"].groupby("noise_sd"):
        noise = g.groupby(["budget_mm", "method"])["yield_t_ha"].mean().rename(f"noise {sd:.0%}")
        t = t.merge(noise.reset_index(), on=["budget_mm", "method"], how="left")
    t["method"] = t["method"].map(lambda m: method_label(plan, m))
    t = t.rename(columns={"budget_mm": "Budget (mm)", "method": "Method", "mean_yield": "Mean",
                          "low_yield": f"{q:.0%} quantile"})
    _write(t, out, "rl" + "r" * (len(t.columns) - 2))


def hyperparameters(plan: Plan, results: Path, out: Path) -> None:
    """The PPO settings actually used, from the first benchmark job of each method and site."""
    settings = table(plan, results, "benchmark", "hyperparameters")
    first = settings.sort_values("job").groupby(["method", "site"]).head(1)
    cols = {f"{method_label(plan, row['method'])} {row['site'].capitalize()}":
            [row[k] for k in PPO_SETTINGS] for _, row in first.iterrows()}
    t = pd.DataFrame(cols, index=PPO_SETTINGS)
    t = t.map(lambda v: f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v))
    t.index.name = "Parameter"
    _write(t.reset_index(), out, "l" * (len(cols) + 1))
