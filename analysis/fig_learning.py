"""Fig: mean validation yield over training steps (E6), median and range over seeds."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from analysis import paper_style as ps
from analysis.common import METHOD_LABEL, derived, save_pdf


def generate(out: Path) -> None:
    df = derived("e1_benchmark_learning")
    cases = df[["site", "budget_mm"]].drop_duplicates().sort_values(["site", "budget_mm"])
    cases = cases.groupby("site").tail(3)  # three budgets per site keep the figure readable
    fig, axes = plt.subplots(1, len(cases), figsize=(2.8 * len(cases), 2.8), squeeze=False)
    for ax, (_, case) in zip(axes[0], cases.iterrows()):
        c = df[(df["site"] == case["site"]) & (df["budget_mm"] == case["budget_mm"])]
        for agent, color in (("drl1", ps.TAB10["blue"]), ("drl2", ps.TAB10["orange"])):
            s = c[c["agent"] == agent].groupby("step")["val_mean_yield"].agg(["median", "min", "max"])
            ax.plot(s.index, s["median"], color=color, label=METHOD_LABEL[agent])
            ax.fill_between(s.index, s["min"], s["max"], color=color, alpha=0.2, lw=0)
        ax.set(title=f"{case['site'].capitalize()}, {case['budget_mm']:g} mm",
               xlabel="Training steps")
        ax.grid(**ps.GRID_KW)
    axes[0][0].set_ylabel("Validation yield (t/ha)")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    save_pdf(fig, out)
