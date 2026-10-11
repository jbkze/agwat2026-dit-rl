"""How the agreed variants learned: validation yield and PPO statistics over the training steps,
and how long each run trained before the stopping rule ended it."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ditrl.analysis import style
from ditrl.report.core import Context, Entry, Section

EXPERIMENTS = ["ablation_one_change", "ablation_more_seeds"]   # the agreed cases, 10 seeds
VARIANTS = list(style.AGREED)
STATISTICS = {   # column of training.csv -> axis label
    "episode_reward": "Episode return",
    "entropy_loss": "Entropy loss",
    "explained_variance": "Explained variance",
    "approx_kl": "Approx. KL",
}
BIN_STEPS = 50_000   # learning curves are averaged within bins of this many steps
# Runs stop at different lengths; a bin is drawn only while at least this share of them has data,
# so that the end of a curve does not rest on one or two long runs.
MIN_RUN_SHARE = 0.5


def _cases(frame: pd.DataFrame) -> list[tuple[str, float]]:
    return sorted(frame[["site", "budget_mm"]].drop_duplicates().itertuples(index=False))


def _median_band(ax, frame: pd.DataFrame, value: str, method: str) -> None:
    binned = frame.assign(step=(frame["step"] // BIN_STEPS + 1) * BIN_STEPS)
    per_run = binned.groupby(["seed", "step"])[value].mean().reset_index()
    runs = per_run["seed"].nunique()
    covered = per_run.groupby("step")["seed"].nunique()
    per_run = per_run[per_run["step"].isin(covered[covered >= MIN_RUN_SHARE * runs].index)]
    s = per_run.groupby("step")[value].quantile([0.25, 0.5, 0.75]).unstack()
    ax.plot(s.index / 1000, s[0.5], color=style.AGREED[method], linewidth=1,
            label=method_label_short(method))
    ax.fill_between(s.index / 1000, s[0.25], s[0.75], color=style.AGREED[method],
                    alpha=style.BAND_ALPHA, linewidth=0)


def method_label_short(method: str) -> str:
    return {"drl1": "DRL1", "drl1_obs": "18 variables", "drl1_reject": "reject",
            "drl1_terminal": "yield at the end"}.get(method, method)


def _restrict(frame: pd.DataFrame) -> pd.DataFrame:
    """Only the agreed variants and their three cases."""
    frame = frame[frame["method"].isin(VARIANTS)]
    agreed_cases = frame[frame["experiment"] == EXPERIMENTS[0]][["site", "budget_mm"]]
    return frame.merge(agreed_cases.drop_duplicates())


def learning_figure(ctx: Context) -> Entry | None:
    curves = ctx.tables_of(EXPERIMENTS, "learning")
    if curves.empty:
        return None
    curves = _restrict(curves)
    cases = _cases(curves)
    fig, axes = plt.subplots(1, len(cases), figsize=(2.3 * len(cases), 2.3), squeeze=False)
    for ax, (site, budget) in zip(axes[0], cases, strict=True):
        c = curves[(curves["site"] == site) & (curves["budget_mm"] == budget)]
        for method in VARIANTS:
            _median_band(ax, c[c["method"] == method], "val_mean_yield", method)
        ax.set_title(f"{site.capitalize()} {budget:g} mm", fontsize=8)
        ax.set_xlabel("Training steps (thousands)")
    axes[0][0].set_ylabel("Validation yield (t/ha)")
    axes[0][0].legend(frameon=False, fontsize=6)
    path = ctx.figures / "agreed_learning_curves"
    style.save(fig, path)
    return Entry(path, "Agreed ablation, learning curves",
                 f"Mean yield on the validation years during training, median and interquartile "
                 f"range over seeds, averaged in bins of {BIN_STEPS:,} steps, drawn while at "
                 f"least half of the runs are still training. In Nebraska the "
                 f"validation years are the test years; nothing is selected on them.", __name__)


def dynamics_figure(ctx: Context) -> Entry | None:
    """PPO statistics over training; runs continued from 300k combine both rounds."""
    current = ctx.tables_of(EXPERIMENTS, "training")
    if current.empty:
        return None
    first = ctx.tables_of(EXPERIMENTS, "training", archive=True)
    if not first.empty:   # the steps before a resume come from the first round
        jobs = ctx.tables_of(EXPERIMENTS, "jobs")[["experiment", "job", "resumed_at"]]
        first = first.merge(jobs, on=["experiment", "job"])
        first = first[first["step"] <= first["resumed_at"]].drop(columns="resumed_at")
        current = pd.concat([first, current], ignore_index=True)
    stats = _restrict(current)
    cases = _cases(stats)
    fig, axes = plt.subplots(len(STATISTICS), len(cases), figsize=(2.3 * len(cases), 1.6 *
                             len(STATISTICS)), sharex=True, squeeze=False)
    for j, (site, budget) in enumerate(cases):
        c = stats[(stats["site"] == site) & (stats["budget_mm"] == budget)]
        for i, (column, name) in enumerate(STATISTICS.items()):
            for method in VARIANTS:
                _median_band(axes[i][j], c[c["method"] == method], column, method)
            if j == 0:
                axes[i][j].set_ylabel(name, fontsize=7)
        axes[0][j].set_title(f"{site.capitalize()} {budget:g} mm", fontsize=8)
        axes[-1][j].set_xlabel("Training steps (thousands)")
    axes[0][0].legend(frameon=False, fontsize=5)
    path = ctx.figures / "agreed_training_dynamics"
    style.save(fig, path)
    return Entry(path, "Agreed ablation, training dynamics",
                 "PPO statistics during training (median and interquartile range over seeds): "
                 "episode return (reward scales differ between rewards), entropy of the policy, "
                 "how much of the return the value function explains, and the size of each "
                 "policy update.", __name__)


def length_figure(ctx: Context) -> Entry | None:
    jobs = ctx.tables_of(ctx.plan.analysis["tables"]["ablation"], "jobs")
    if jobs.empty or "steps_trained" not in jobs:
        return None
    jobs = jobs[jobs["method"].isin(VARIANTS)].dropna(subset=["steps_trained"])
    cases = _cases(jobs)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    width = 0.8 / len(VARIANTS)
    for x, (site, budget) in enumerate(cases):
        for k, method in enumerate(VARIANTS):
            g = jobs[(jobs["site"] == site) & (jobs["budget_mm"] == budget)
                     & (jobs["method"] == method)]
            ax.scatter([x - 0.4 + width * (k + 0.5)] * len(g), g["steps_trained"] / 1000, s=5,
                       alpha=0.5, color=style.AGREED[method],
                       label=method_label_short(method) if x == 0 else None)
    ax.set_xticks(range(len(cases)), [f"{s.capitalize()}\n{b:g} mm" for s, b in cases])
    ax.set_ylabel("Steps trained (thousands)")
    ax.legend(frameon=False, fontsize=6)
    path = ctx.figures / "training_length"
    style.save(fig, path)
    return Entry(path, "Training length",
                 "Steps each run of the agreed variants trained before the stopping rule ended "
                 "it, one dot per run.", __name__)


def build(ctx: Context) -> Section:
    section = Section("Appendix: training diagnostics")
    for make in (learning_figure, dynamics_figure, length_figure):
        entry = make(ctx)
        if entry:
            section.entries.append(entry)
    return section

