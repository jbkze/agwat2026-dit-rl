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
SITE_COLORS = {"nebraska": style.BRIGHT["blue"], "tunis": style.BRIGHT["red"]}
PENDING_COLOR = "0.6"
COLUMNS = 3          # panels per row of the figure
SITE_GAP = 1         # empty slots between the sites on the x axis


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


def _slots(table: pd.DataFrame) -> dict[tuple[str, float], int]:
    """x position of every case: the sites one after the other, with a gap between them."""
    slots, x = {}, 0
    for site in sorted(table["site"].unique()):
        for budget in sorted(table.loc[table["site"] == site, "budget_mm"].unique()):
            slots[(site, budget)] = x
            x += 1
        x += SITE_GAP
    return slots


def figure(ctx: Context, table: pd.DataFrame) -> Entry:
    questions = list(dict.fromkeys(table["question"]))
    slots = _slots(table)
    rows = math.ceil(len(questions) / COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(2.4 * COLUMNS, 1.55 * rows), sharex=True,
                             sharey=True, squeeze=False)
    for ax, qid in zip(axes.flat, questions):
        q = table[table["question"] == qid]
        effect, test = q[q["kind"] == "effect"], q[q["kind"] == "check"]
        for _, r in effect.iterrows():
            x = slots[(r["site"], r["budget_mm"])]
            if math.isnan(r["effect"]):
                ax.plot(x, 0, marker="x", markersize=3, color=PENDING_COLOR, linestyle="none")
                continue
            clear = r["ci_low"] > 0 or r["ci_high"] < 0
            color = SITE_COLORS.get(r["site"])
            ax.errorbar(x, r["effect"], yerr=[[r["effect"] - r["ci_low"]], [r["ci_high"] - r["effect"]]],
                        marker="o", markersize=3, capsize=1.5, linewidth=0.8, color=color,
                        markerfacecolor=color if clear else "white")
        for _, r in test.dropna(subset=["effect"]).iterrows():
            ax.plot(slots[(r["site"], r["budget_mm"])] + 0.3, r["effect"], marker="D",
                    markersize=2.2, color="k", linestyle="none")
        if effect["effect"].isna().all():
            ax.text(0.5, 0.5, f"no results yet\n({int(effect['done'].sum())} of "
                    f"{int(effect['planned'].sum())} runs)", transform=ax.transAxes, ha="center",
                    va="center", fontsize=6, color=PENDING_COLOR)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_title(f"{q['group'].iloc[0]}: {q['change'].iloc[0]}", fontsize=6.5)
    for ax in axes.flat[len(questions):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xticks(list(slots.values()), [f"{b:g}" for _, b in slots], fontsize=5, rotation=90)
        for label, (site, _) in zip(ax.get_xticklabels(), slots, strict=True):
            label.set_color(SITE_COLORS.get(site))
    for ax in axes[:, 0]:
        ax.set_ylabel("Change (t/ha)", fontsize=6)
    handles = [plt.Line2D([], [], marker="o", color=c, linestyle="none", markersize=3,
                          label=s.capitalize()) for s, c in SITE_COLORS.items()]
    handles += [plt.Line2D([], [], marker="o", color="k", markerfacecolor="white",
                           linestyle="none", markersize=3, label="interval contains 0"),
                plt.Line2D([], [], marker="D", color="k", linestyle="none", markersize=2.2,
                           label="averaged over the other choices"),
                plt.Line2D([], [], marker="x", color=PENDING_COLOR, linestyle="none",
                           markersize=3, label="planned, results pending")]
    fig.supxlabel("Seasonal budget (mm), Nebraska in blue, Tunis in red", fontsize=7)
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, 0.0))
    path = ctx.figures / "sensitivity"
    style.save(fig, path)
    return Entry(path, "What changes what",
                 "Each panel changes one thing about DRL1. Dot: mean change of test yield against "
                 "DRL1 trained with the same seed, with the 95 % bootstrap interval over seeds, "
                 "filled when the interval does not contain 0. Diamond: the same change averaged over the other "
                 "choices of the 2 x 4 x 2 design. Cross: planned, results pending.", __name__)


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
    ctx.facts["sensitivity"] = table
    path = ctx.tables / "sensitivity.md"
    write_table(summary(table), path)
    table.to_csv(ctx.tables / "sensitivity_long.csv", index=False)
    return Section("What changes what", [
        figure(ctx, table),
        Entry(path, "What changes what (table)",
              "Change of test yield in t/ha against DRL1 with the 95 % interval, per case. "
              "\"(k of n)\": k of the n planned runs have results. \"pending\": too few seeds "
              "have results yet. Empty: not planned. The file `sensitivity_long.csv` holds every "
              "number.", __name__)])
