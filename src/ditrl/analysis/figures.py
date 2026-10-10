"""Paper figures. Each generator reads the result tables, writes one PDF and, where useful, a
CSV with the plotted numbers next to it."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ditrl.analysis import style
from ditrl.analysis.data import method_label, oracle, table, test_seasons
from ditrl.analysis.stats import paired_differences
from ditrl.paths import REPO
from ditrl.plan import Plan


def _site_axes(sites: list[str], height: float = style.PANEL_HEIGHT):
    fig, axes = plt.subplots(1, len(sites), figsize=(style.PANEL_WIDTH * len(sites), height),
                             squeeze=False)
    return fig, axes[0]


def cwpf(plan: Plan, results: Path, out: Path) -> None:
    """Crop water production function per site (E1): mean test yield over the budget. DRL lines
    show the median over seeds, the band spans the worst to the best seed."""
    df = test_seasons(plan, results, "benchmark")
    per_seed = (df.groupby(["site", "method", "budget_mm", "seed"], dropna=False)["yield_t_ha"]
                .mean().reset_index())
    ceiling = (oracle(plan, results, "ds").groupby(["site", "budget_mm"])["yield_t_ha"].mean()
               .reset_index())
    external = REPO / plan.analysis["external_heuristics"]
    sites = sorted(df["site"].unique())
    fig, axes = _site_axes(sites)
    for ax, site in zip(axes, sites, strict=True):
        o = ceiling[ceiling["site"] == site]
        ax.plot(o["budget_mm"], o["yield_t_ha"], "k--", lw=1.2, label="EA-DS (perfect foresight)")
        for method, (name, color) in style.METHODS.items():
            g = per_seed[(per_seed["site"] == site) & (per_seed["method"] == method)]
            if g.empty:
                continue
            if g["budget_mm"].isna().all():   # no budget: one horizontal reference line
                ax.axhline(g["yield_t_ha"].iloc[0], color=color, lw=1, ls=":", label=name)
                continue
            s = g.groupby("budget_mm")["yield_t_ha"].agg(["median", "min", "max"]).reset_index()
            ax.plot(s["budget_mm"], s["median"], marker="o", ms=3, color=color, label=name)
            ax.fill_between(s["budget_mm"], s["min"], s["max"], color=color,
                            alpha=style.BAND_ALPHA, lw=0)
        if external.exists():   # Simplex rules of the co-author, if delivered
            ext = pd.read_csv(external)
            ext = ext[ext["site"] == site].groupby(["method", "budget_mm"])["yield_t_ha"].mean()
            for method, s in ext.groupby(level=0):
                ax.plot(s.index.get_level_values(1), s.values, ls="-.", lw=1, label=method)
        ax.set(title=site.capitalize(), xlabel="Seasonal budget (mm)", ylabel="Yield (t/ha)")
        ax.grid(**style.GRID)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    per_seed.to_csv(out.with_suffix(".csv"), index=False)
    style.save(fig, out.with_suffix(""))


def ablation(plan: Plan, results: Path, out: Path) -> None:
    """Ablation (E2): for every changed design and case, the paired difference of mean test yield
    to the reference method, one dot per seed and a bar at their mean. Positive values mean that
    the change costs yield."""
    cfg = plan.analysis
    diffs = paired_differences(test_seasons(plan, results, "ablation"), cfg["ablation_reference"],
                               cfg["lower_quantile"])
    cases = diffs[["site", "budget_mm"]].drop_duplicates().sort_values(["site", "budget_mm"])
    methods = sorted(diffs["method"].unique())
    size = (style.PANEL_WIDTH * len(cases), style.PANEL_HEIGHT)
    fig, axes = plt.subplots(1, len(cases), figsize=size, squeeze=False, sharey=True)
    for ax, (_, case) in zip(axes[0], cases.iterrows(), strict=True):
        c = diffs[(diffs["site"] == case["site"]) & (diffs["budget_mm"] == case["budget_mm"])]
        for x, method in enumerate(methods):
            d = c[c["method"] == method]["mean_difference"]
            ax.plot([x] * len(d), d, "o", ms=4, color=style.TAB10["blue"], alpha=style.DOT_ALPHA)
            half = style.MEAN_BAR_HALF_WIDTH
            ax.hlines(d.mean(), x - half, x + half, color="k", lw=1.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(methods)), [method_label(plan, m) for m in methods],
                      rotation=style.TICK_ROTATION, ha="right")
        ax.set(title=f"{case['site'].capitalize()}, {case['budget_mm']:g} mm")
        ax.grid(axis="y", **style.GRID)
    reference = method_label(plan, cfg["ablation_reference"])
    axes[0][0].set_ylabel(f"Yield of {reference} minus variant (t/ha)")
    fig.tight_layout()
    diffs.to_csv(out.with_suffix(".csv"), index=False)
    style.save(fig, out.with_suffix(""))


def actions_low_budget(plan: Plan, results: Path, out: Path) -> None:
    """Which depths DRL1 applies at low budgets in Tunis (E4). Top: share of all decisions per
    applied depth. Bottom: median remaining budget over the season."""
    cfg = plan.analysis["figures"]["actions_low_budget"]
    d = table(plan, results, "low_budget", "decisions")
    d = d[d["budget_mm"].isin(cfg["budgets"]) & d["site"].eq(cfg["site"])
          & d["method"].isin(cfg["methods"])]
    colors = list(style.TAB10.values())
    width, height = style.TALL_PANEL
    fig, axes = plt.subplots(2, len(cfg["budgets"]), figsize=(width * len(cfg["budgets"]), height),
                             sharey="row", squeeze=False)
    for j, budget in enumerate(cfg["budgets"]):
        b = d[d["budget_mm"] == budget]
        for method, color in zip(cfg["methods"], colors, strict=False):
            m = b[b["method"] == method]
            share = m["applied_mm"].round().value_counts(normalize=True).sort_index()
            axes[0, j].plot(share.index, share.values, marker="o", ms=3, color=color,
                            label=method_label(plan, method))
            curve = m.groupby("dap")["remaining_mm"].median()
            axes[1, j].plot(curve.index, curve.values, color=color)
        axes[0, j].set(title=f"{budget} mm", xlabel="Applied depth (mm)")
        axes[1, j].set(xlabel="Days after planting")
        for ax in axes[:, j]:
            ax.grid(**style.GRID)
    axes[0, 0].set_ylabel("Share of decisions")
    axes[1, 0].set_ylabel("Remaining budget (mm)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    style.save(fig, out.with_suffix(""))


def learning_curves(plan: Plan, results: Path, out: Path) -> None:
    """Mean validation yield over training steps (E6): median and range over seeds."""
    cfg = plan.analysis["figures"]["learning_curves"]
    df = table(plan, results, "benchmark", "learning")
    cases = df[["site", "budget_mm"]].drop_duplicates().sort_values(["site", "budget_mm"])
    cases = cases.groupby("site").tail(cfg["budgets_per_site"])
    size = (style.SMALL_PANEL * len(cases), style.SMALL_PANEL)
    fig, axes = plt.subplots(1, len(cases), figsize=size, squeeze=False)
    for ax, (_, case) in zip(axes[0], cases.iterrows(), strict=True):
        c = df[(df["site"] == case["site"]) & (df["budget_mm"] == case["budget_mm"])]
        for method in cfg["methods"]:
            name, color = style.METHODS[method]
            s = (c[c["method"] == method].groupby("step")["val_mean_yield"]
                 .agg(["median", "min", "max"]))
            ax.plot(s.index, s["median"], color=color, label=name)
            ax.fill_between(s.index, s["min"], s["max"], color=color, alpha=style.BAND_ALPHA, lw=0)
        ax.set(title=f"{case['site'].capitalize()}, {case['budget_mm']:g} mm",
               xlabel="Training steps")
        ax.grid(**style.GRID)
    axes[0][0].set_ylabel("Validation yield (t/ha)")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    style.save(fig, out.with_suffix(""))
