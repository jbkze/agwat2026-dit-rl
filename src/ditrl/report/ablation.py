"""The full ablation (2 observations x 4 budget handlings x 2 rewards): main effect of each design
choice over the budgets, and what training until convergence changed against 300,000 steps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ditrl.analysis import style
from ditrl.analysis.stats import main_effects, per_policy
from ditrl.report.core import Context, Entry, Section, write_table

SITE_COLORS = {"nebraska": style.BRIGHT["blue"], "tunis": style.BRIGHT["red"]}
SIGNIFICANCE = 0.05


def _effects(ctx: Context, seasons: pd.DataFrame) -> pd.DataFrame:
    cfg = ctx.plan.analysis
    site = next(iter(ctx.plan.sites))
    factors = {f: list(levels) for f, levels in cfg["ablation_factors"].items()}
    reference = {f: str(ctx.plan.method(cfg["ablation_reference"], site)[f]) for f in factors}
    table = main_effects(seasons, factors, reference, cfg["bootstrap"], cfg["wilcoxon_min_pairs"])
    table["change"] = [cfg["ablation_factors"][f][lvl]
                       for f, lvl in zip(table["factor"], table["level"], strict=True)]
    return table


def effects_figure(ctx: Context, table: pd.DataFrame) -> Entry:
    changes = list(dict.fromkeys(table["change"]))
    fig, axes = plt.subplots(1, len(changes), figsize=(1.9 * len(changes), 2.3), sharey=True,
                             squeeze=False)
    for ax, change in zip(axes[0], changes, strict=True):
        for site, g in table[table["change"] == change].groupby("site"):
            g = g.sort_values("budget_mm")
            ax.errorbar(g["budget_mm"], g["effect_t_ha"],
                        yerr=[g["effect_t_ha"] - g["ci_low"], g["ci_high"] - g["effect_t_ha"]],
                        marker="o", markersize=3, capsize=2, linewidth=1,
                        color=SITE_COLORS.get(site), label=site.capitalize())
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_title(change, fontsize=8)
        ax.set_xlabel("Budget (mm)")
    axes[0][0].set_ylabel("Yield change vs. DRL1 (t/ha)")
    axes[0][0].legend(frameon=False, fontsize=6)
    path = ctx.figures / "ablation_main_effects"
    style.save(fig, path)
    return Entry(path, "Ablation, main effects",
                 "Change of mean test yield when one design choice of DRL1 is switched, paired "
                 "over the other choices, seeds and test years, with 95 % bootstrap intervals; "
                 "one line per site over the seasonal budget.", __name__)


def convergence_figure(ctx: Context, experiments: list[str]) -> Entry | None:
    """Paired change of test yield per run: trained until convergence minus 300,000 steps."""
    q = ctx.plan.analysis["lower_quantile"]
    new = ctx.tables_of(experiments, "seasons")
    old = ctx.tables_of(experiments, "seasons", archive=True)
    if new.empty or old.empty:
        return None
    keys = ["method", "site", "budget_mm", "seed"]
    pairs = per_policy(new[new["split"] == "test"], q).merge(
        per_policy(old[old["split"] == "test"], q), on=keys, suffixes=("", "_fixed"))
    pairs["gain"] = pairs["mean_yield"] - pairs["mean_yield_fixed"]
    cases = pairs[["site", "budget_mm"]].drop_duplicates().sort_values(["site", "budget_mm"])
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for x, (_, case) in enumerate(cases.iterrows()):
        g = pairs[(pairs["site"] == case["site"]) & (pairs["budget_mm"] == case["budget_mm"])]
        jitter = (g["seed"] % 10 - 4.5) * 0.03
        ax.scatter(x + jitter, g["gain"], s=5, alpha=0.4, color=SITE_COLORS.get(case["site"]))
        ax.hlines(g["gain"].mean(), x - 0.3, x + 0.3, color="k", linewidth=1.2)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xticks(range(len(cases)), [f"{s.capitalize()}\n{b:g} mm" for s, b in
                                      zip(cases["site"], cases["budget_mm"], strict=True)])
    ax.set_ylabel("Converged minus 300k (t/ha)")
    path = ctx.figures / "convergence_gain"
    style.save(fig, path)
    return Entry(path, "What training until convergence changed",
                 f"Test yield of each run trained until convergence minus the same run after "
                 f"300,000 steps (same variant, case and seed; {len(pairs)} runs), bar at the "
                 f"mean.", __name__)


def build(ctx: Context) -> Section:
    experiments = ctx.plan.analysis["tables"]["ablation"]
    seasons = ctx.tables_of(experiments, "seasons")
    if seasons.empty:
        return Section()
    seasons = seasons[seasons["split"] == "test"]
    section = Section()
    table = _effects(ctx, seasons)
    path = ctx.tables / "ablation_main_effects.md"
    shown = table[["change", "site", "budget_mm", "effect_t_ha", "ci_low", "ci_high", "pairs",
                   "p_holm"]].rename(columns={"change": "Changed to", "site": "Site",
                                              "budget_mm": "Budget (mm)",
                                              "effect_t_ha": "Effect (t/ha)",
                                              "ci_low": "95 % low", "ci_high": "95 % high",
                                              "pairs": "Pairs", "p_holm": "p (Holm)"})
    write_table(shown, path, decimals=3)
    section.entries += [Entry(path, "Ablation, main effects (table)",
                              "The numbers behind the main-effects figure.", __name__),
                        effects_figure(ctx, table)]
    gain = convergence_figure(ctx, experiments)
    if gain:
        section.entries.append(gain)
    planned = sum(len(ctx.plan.expand(e)) for e in experiments if e in ctx.plan.experiments)
    finished = len(ctx.tables_of(experiments, "jobs"))
    if finished < planned:
        section.findings.append(f"Ablation so far: {finished} of {planned} runs; effects at "
                                "budgets with few runs are preliminary.")
    significant = table[table["p_holm"] < SIGNIFICANCE]
    for change, g in significant.groupby("change", sort=False):
        cases = ", ".join(f"{r.site.capitalize()} {r.budget_mm:g} mm {r.effect_t_ha:+.2f}"
                          for r in g.sort_values(["site", "budget_mm"]).itertuples())
        section.findings.append(f"Switching to {change} changes test yield significantly in: "
                                f"{cases} t/ha.")
    return section
