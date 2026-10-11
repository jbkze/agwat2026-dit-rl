"""The ablation agreed with the co-author: DRL1 and three variants that change one design choice
each, on three cases. Configuration table, result table and the paired differences per seed, as
the agreed plan asks, over all seeds (the agreed five and the five extra ones)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ditrl.analysis import style
from ditrl.analysis.data import method_label
from ditrl.analysis.stats import paired_differences, per_policy
from ditrl.report.core import Context, Entry, Section, write_table

EXPERIMENT = "ablation_one_change"
EXTRA_SEEDS = "ablation_more_seeds"
VARIANTS = list(style.AGREED)            # DRL1 first: the reference
PPO_SHOWN = ["algorithm", "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
             "gae_lambda", "clip_range", "ent_coef", "net_arch"]


def _case(site: str, budget: float) -> str:
    return f"{site.capitalize()} {budget:g} mm"


def configuration(ctx: Context) -> pd.DataFrame:
    """One column per variant: the design choices, the PPO settings actually used, training."""
    plan, site = ctx.plan, next(iter(ctx.plan.sites))
    factors = plan.analysis["ablation_factors"]
    settings = ctx.table(EXPERIMENT, "hyperparameters")
    rows = {"Observation": [], "Budget handling": [], "Reward": [], "Action set (mm)": [],
            "Decision interval (days)": []}
    for method in VARIANTS:
        cfg = plan.method(method, site)
        rows["Observation"].append(factors["observation"][cfg["observation"]])
        rows["Budget handling"].append(factors["constraint"][cfg["constraint"]])
        rows["Reward"].append(factors["reward"][cfg["reward"]])
        rows["Action set (mm)"].append(", ".join(f"{d:g}" for d in cfg["depths"]))
        rows["Decision interval (days)"].append(str(cfg["interval"]))
    if settings is not None:
        first = settings.sort_values("job").groupby("method").head(1).set_index("method")
        for key in PPO_SHOWN:
            rows[f"PPO {key}"] = [str(first.loc[m, key]) if m in first.index else ""
                                  for m in VARIANTS]
    rule = plan.method("drl1", site)["training"]
    rows["Training"] = [f"until the mean yield of the training seasons rises by no more than "
                        f"{rule['tolerance']} t/ha in {rule['patience']} blocks of "
                        f"{rule['block_steps']:,} steps, {rule['min_steps']:,} to "
                        f"{rule['max_steps']:,} steps"] * len(VARIANTS)
    rows["Evaluation"] = ["final policy, deterministic actions, test years"] * len(VARIANTS)
    table = pd.DataFrame(rows, index=[method_label(plan, m) for m in VARIANTS]).T
    return table.reset_index(names="Setting")


def results(ctx: Context, seasons: pd.DataFrame) -> pd.DataFrame:
    """Per case and variant: test yield over seeds and the paired difference to DRL1."""
    q = ctx.plan.analysis["lower_quantile"]
    policies = per_policy(seasons, q)
    diffs = paired_differences(seasons, VARIANTS[0], q)
    rows = []
    for (site, budget), g in policies.groupby(["site", "budget_mm"]):
        for method in VARIANTS:
            p = g[g["method"] == method]
            d = diffs[(diffs["site"] == site) & (diffs["budget_mm"] == budget)
                      & (diffs["method"] == method)]
            several_years = p["years"].max() > 1
            rows.append({
                "Case": _case(site, budget), "Variant": method_label(ctx.plan, method),
                "Seeds": len(p), "Mean yield (t/ha)": p["mean_yield"].mean(),
                "SD over seeds": p["mean_yield"].std(),
                "DRL1 minus variant: mean": d["mean_difference"].mean() if len(d) else None,
                "lowest": d["mean_difference"].min() if len(d) else None,
                "highest": d["mean_difference"].max() if len(d) else None,
                f"{q:.0%} quantile (t/ha)": p["low_yield"].mean() if several_years else None,
                "Note": ", ".join(n for n in (
                    "reference (no difference)" if method == VARIANTS[0] else "",
                    "one test year (no quantile over years)" if not several_years else "") if n),
            })
    return pd.DataFrame(rows)


def paired_figure(ctx: Context, seasons: pd.DataFrame, name: str) -> Entry:
    q = ctx.plan.analysis["lower_quantile"]
    diffs = paired_differences(seasons, VARIANTS[0], q)
    cases = diffs[["site", "budget_mm"]].drop_duplicates().sort_values(["site", "budget_mm"])
    seeds = diffs["seed"].nunique()
    fig, axes = plt.subplots(1, len(cases), figsize=(2.3 * len(cases), 2.4), sharey=True,
                             squeeze=False)
    for ax, (_, case) in zip(axes[0], cases.iterrows(), strict=True):
        c = diffs[(diffs["site"] == case["site"]) & (diffs["budget_mm"] == case["budget_mm"])]
        for x, method in enumerate(VARIANTS[1:]):
            d = c[c["method"] == method]["mean_difference"]
            ax.scatter([x] * len(d), d, s=12, color=style.AGREED[method], alpha=style.DOT_ALPHA,
                       zorder=3)
            ax.hlines(d.mean(), x - style.MEAN_BAR_HALF_WIDTH, x + style.MEAN_BAR_HALF_WIDTH,
                      color="k", linewidth=1.2, zorder=4)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_xticks(range(len(VARIANTS) - 1),
                      [method_label(ctx.plan, m).removeprefix("DRL1, ") for m in VARIANTS[1:]],
                      rotation=style.TICK_ROTATION, ha="right")
        ax.set_xlim(-0.6, len(VARIANTS) - 1.4)
        ax.set_title(_case(case["site"], case["budget_mm"]), fontsize=8)
    axes[0][0].set_ylabel("DRL1 minus variant (t/ha)")
    path = ctx.figures / name
    style.save(fig, path)
    return Entry(path, f"Core ablation, paired differences ({seeds} seeds)",
                 f"Mean test yield of DRL1 minus that of each variant, one dot per seed "
                 f"({seeds} seeds), bar at the mean. Above zero: the change costs yield; below "
                 f"zero: the variant is better than DRL1.", __name__)


def build(ctx: Context) -> Section:
    base = ctx.table(EXPERIMENT, "seasons")
    if base is None:
        return Section()
    base = base[(base["split"] == "test") & base["method"].isin(VARIANTS)]
    section = Section("Core ablation")
    config_path = ctx.tables / "core_ablation_configuration.md"
    write_table(configuration(ctx), config_path)
    section.entries.append(Entry(config_path, "Core ablation, configuration",
                                 "Design choices, PPO settings actually used, training and "
                                 "evaluation of the four variants.", __name__))
    seasons = base
    extra = ctx.table(EXTRA_SEEDS, "seasons")
    if extra is not None:   # the extra seeds, on the agreed cases only
        extra = extra[(extra["split"] == "test") & extra["method"].isin(VARIANTS)]
        seasons = pd.concat([base, extra], ignore_index=True).merge(
            base[["site", "budget_mm"]].drop_duplicates())
    path = ctx.tables / "core_ablation_results.md"
    write_table(results(ctx, seasons), path)
    section.entries += [Entry(path, "Core ablation, results",
                              "Test yield per case and variant over all seeds, and the paired "
                              "difference to DRL1 per seed.", __name__),
                        paired_figure(ctx, seasons, "core_ablation")]
    return section
