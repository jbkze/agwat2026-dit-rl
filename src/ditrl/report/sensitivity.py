"""What changes what: every design choice, information source and disturbance of the plan is
examined the same way, as one change away from DRL1 (config/plan.yaml, analysis.sensitivity).

    effect   runs of the variant are paired with DRL1 runs of the same site, budget and training
             seed on the same test years; per seed the mean yield difference over the test years,
             variant minus DRL1; the effect is the mean over seeds with a bootstrap interval over
             seeds. Positive: the change raises yield.
    check    for the three factors of the 2 x 4 x 2 design, the same change averaged over all
             levels of the other two factors. Far from the effect: the factors interact.

Cases the plan contains but that have too few results yet stay in the figure and the table as
placeholders, so that the gaps are visible.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd

from ditrl.analysis import style
from ditrl.analysis.stats import bootstrap_ci
from ditrl.report.core import Context, Entry, Section, write_table

PAIR = ["site", "budget_mm", "seed", "year"]
CASE = ["site", "budget_mm"]
TRAINING_SEED = r"_s(\d+)_noise"   # evaluation jobs: the seed of the trained model is in the id
COLORS = [style.BRIGHT["yellow"], style.BRIGHT["green"], style.BRIGHT["red"],
          style.BRIGHT["purple"], style.BRIGHT["cyan"]]   # one per shown question
SITE_NAMES = {"nebraska": "Nebraska (rain)", "tunis": "Tunis (no rain)"}


def _test(ctx: Context, experiments: list[str], archive: bool = False) -> pd.DataFrame:
    seasons = ctx.tables_of(experiments, "seasons", archive)
    return seasons[seasons["split"] == "test"] if not seasons.empty else seasons


def _experiments(cfg: dict, q: dict) -> list[str]:
    return [q["experiment"]] if "experiment" in q else cfg["trainings"]


def _variant(ctx: Context, cfg: dict, q: dict) -> pd.DataFrame:
    seasons = _test(ctx, _experiments(cfg, q), q.get("archive", False))
    if seasons.empty:
        return seasons
    seasons = seasons[seasons["method"] == q["method"]]
    if "noise_sd" in q:
        seasons = seasons[seasons["noise_sd"] == q["noise_sd"]]
        seasons = seasons.assign(seed=seasons["job"].str.extract(TRAINING_SEED)[0].astype(int))
    return seasons


def _summarise(diffs: pd.DataFrame, units: list[str], cfg: dict, boot: dict) -> pd.DataFrame:
    """Rows with a `diff` per paired season -> mean per unit -> effect per case over units."""
    per_unit = diffs.groupby(CASE + units)["diff"].mean().reset_index()
    rows = []
    for (site, budget), g in per_unit.groupby(CASE):
        enough = len(g) >= cfg["min_seeds"]
        low, high = bootstrap_ci(g["diff"], **boot) if enough else (math.nan, math.nan)
        rows.append({"site": site, "budget_mm": budget, "units": len(g),
                     "effect": g["diff"].mean() if enough else math.nan,
                     "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows, columns=CASE + ["units", "effect", "ci_low", "ci_high"])


def one_change(ctx: Context, cfg: dict, q: dict, reference: pd.DataFrame) -> pd.DataFrame:
    variant = _variant(ctx, cfg, q)
    if variant.empty or reference.empty:
        return _summarise(pd.DataFrame(columns=PAIR + ["diff"]), ["seed"], cfg, {})
    pairs = variant.groupby(PAIR, as_index=False)["yield_t_ha"].mean().merge(
        reference, on=PAIR, suffixes=("", "_reference"))
    pairs["diff"] = pairs["yield_t_ha"] - pairs["yield_t_ha_reference"]
    return _summarise(pairs, ["seed"], cfg, ctx.plan.analysis["bootstrap"])


def check(ctx: Context, cfg: dict, q: dict) -> pd.DataFrame | None:
    """The change of one factor, averaged over the levels of the other factors of the design."""
    if "factor" not in q:
        return None
    seasons = _test(ctx, cfg["design"])
    if seasons.empty:
        return None
    site = next(iter(ctx.plan.sites))
    factor, factors = q["factor"], list(ctx.plan.analysis["ablation_factors"])
    level = str(ctx.plan.method(q["method"], site)[factor])
    base = str(ctx.plan.method(cfg["reference"], site)[factor])
    others = [f for f in factors if f != factor]
    wide = seasons.pivot_table(index=PAIR + others, columns=factor, values="yield_t_ha")
    if level not in wide or base not in wide:
        return None
    diffs = (wide[level] - wide[base]).dropna().rename("diff").reset_index()
    return _summarise(diffs, ["seed", *others], cfg, ctx.plan.analysis["bootstrap"])


def counts(ctx: Context, cfg: dict, q: dict) -> pd.DataFrame:
    """Planned and finished runs of a question per case."""
    method, noise = q["method"], q.get("noise_sd")
    if q.get("archive"):
        jobs = ctx.tables_of(cfg["trainings"], "jobs", archive=True)
        jobs = jobs[jobs["method"] == method] if not jobs.empty else jobs
        size = jobs.groupby(CASE).size() if not jobs.empty else pd.Series(dtype=int)
        return pd.DataFrame({"planned": size, "done": size}).reset_index()
    planned = pd.DataFrame(
        [(j.site, j.budget) for e in _experiments(cfg, q) for j in ctx.plan.expand(e)
         if j.method == method and next(iter((j.config.get("noise") or {}).values()), None) == noise],
        columns=CASE).value_counts().rename("planned")
    jobs = ctx.tables_of(_experiments(cfg, q), "jobs")
    if not jobs.empty:
        jobs = jobs[(jobs["method"] == method)
                    & (jobs["noise_sd"] == noise if noise is not None else True)]
    table = planned.to_frame()
    table["done"] = jobs.groupby(CASE).size() if not jobs.empty else 0
    return table.reset_index().fillna({"done": 0}).astype({"done": int})


def analyse(ctx: Context) -> pd.DataFrame:
    """One row per question, case and kind (effect or check), with planned and finished runs."""
    cfg = ctx.plan.analysis["sensitivity"]
    reference = _test(ctx, cfg["trainings"])
    if not reference.empty:
        reference = (reference[reference["method"] == cfg["reference"]]
                     .groupby(PAIR, as_index=False)["yield_t_ha"].mean())
    rows = []
    for qid, q in cfg["questions"].items():
        n = counts(ctx, cfg, q)
        for kind, result in (("effect", one_change(ctx, cfg, q, reference)),
                             ("check", check(ctx, cfg, q))):
            if result is None:
                continue
            frame = n.merge(result, on=CASE, how="left") if kind == "effect" else \
                result.merge(n[CASE], on=CASE)
            rows.append(frame.assign(question=qid, group=q["group"], change=q["change"],
                                     kind=kind))
    return pd.concat([r for r in rows if not r.empty], ignore_index=True)


def figure(ctx: Context, table: pd.DataFrame) -> Entry:
    """The questions listed under `shown`, as lines over the budget per site. Budgets without
    results stay on the axis and leave a gap in the line."""
    shown = ctx.plan.analysis["sensitivity"]["shown"]
    questions = ctx.plan.analysis["sensitivity"]["questions"]
    sites = sorted(table["site"].unique())
    budgets = {s: sorted(table.loc[table["site"] == s, "budget_mm"].unique()) for s in sites}
    fig, axes = plt.subplots(len(shown), len(sites), figsize=(6.6, 2.05 * len(shown)),
                             sharey="row", squeeze=False,
                             gridspec_kw={"width_ratios": [len(budgets[s]) + 2 for s in sites]})
    for i, (qid, color) in enumerate(zip(shown, COLORS, strict=False)):
        q = questions[qid]
        for j, site in enumerate(sites):
            ax = axes[i][j]
            grid = pd.DataFrame({"site": site, "budget_mm": budgets[site]})
            sel = table[table["question"] == qid]
            one = grid.merge(sel[sel["kind"] == "effect"], how="left")
            avg = grid.merge(sel[sel["kind"] == "check"], how="left")
            ax.fill_between(one["budget_mm"], one["ci_low"], one["ci_high"], color=color,
                            alpha=style.BAND_ALPHA, linewidth=0)
            ax.plot(one["budget_mm"], one["effect"], color=color, marker="o", markersize=2.5,
                    linewidth=1.1, label="single change")
            if avg["effect"].notna().any():
                ax.plot(avg["budget_mm"], avg["effect"], color=color, linestyle="--",
                        linewidth=0.9, label="averaged over the other choices")
            ax.axhline(0, color="k", linewidth=0.5)
            ax.set_xticks(budgets[site])
            ax.tick_params(axis="x", labelsize=6)
            margin = 0.04 * (budgets[site][-1] - budgets[site][0])
            ax.set_xlim(budgets[site][0] - margin, budgets[site][-1] + margin)
            if i == 0:
                ax.set_title(SITE_NAMES.get(site, site), fontsize=8)
            if i == len(shown) - 1:
                ax.set_xlabel("Seasonal budget (mm)", fontsize=7)
        axes[i][0].set_ylabel("Change against DRL1 (t/ha)", fontsize=6.5)
        axes[i][-1].text(0.01, 0.97, f"{q['group']}: {q['change']}", transform=axes[i][-1].transAxes,
                         fontsize=7, va="top", fontweight="bold")
        axes[i][-1].legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout()
    path = ctx.figures / "effects_over_budget"
    style.save(fig, path)
    return Entry(path, "First look: effect of three design choices over the budget",
                 "Test yield of the variant minus DRL1 with the same seed, mean over seeds with "
                 "95 % bootstrap band (solid), and the same change averaged over the other choices "
                 "of the 2 x 4 x 2 design (dashed). Gaps: no results for that budget.", __name__)


def _cell(r) -> str:
    if math.isnan(r["effect"]):
        if r["kind"] == "check":
            return "pending"
        return f"pending ({int(r['done'])} of {int(r['planned'])})"
    text = f"{r['effect']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
    if r["kind"] == "effect" and r["done"] < r["planned"]:
        text += f" ({int(r['done'])} of {int(r['planned'])})"
    return text


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """The figure as a table: one row per question (and check), one column per case."""
    shown = table.assign(cell=table.apply(_cell, axis=1),
                         case=[f"{s.capitalize()} {b:g}" for s, b in
                               zip(table["site"], table["budget_mm"], strict=True)],
                         row=[c if k == "effect" else f"{c}, averaged over the other choices"
                              for c, k in zip(table["change"], table["kind"], strict=True)])
    order = list(dict.fromkeys(shown.sort_values(CASE)["case"]))
    wide = shown.pivot_table(index=["group", "row"], columns="case", values="cell",
                             aggfunc="first", sort=False)
    return wide.reindex(columns=order).fillna("").reset_index().rename(
        columns={"group": "Group", "row": "Change"})


def build(ctx: Context) -> Section:
    table = analyse(ctx)
    path = ctx.tables / "sensitivity.md"
    write_table(summary(table), path)
    table.to_csv(ctx.tables / "sensitivity_long.csv", index=False)
    return Section("First look", [
        figure(ctx, table),
        Entry(path, "Every question of the plan (table)",
              "Change of test yield in t/ha against DRL1 with the 95 % interval, per case. "
              "\"(k of n)\": k of the n planned runs have results. \"pending\": too few seeds "
              "have results yet. Empty: not planned.", __name__)])
