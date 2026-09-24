"""Fig: which depths DRL1 applies at low budgets in Tunis (E4, R3#3). Top row: share of all
decisions per applied depth. Bottom row: median remaining budget over the season."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from analysis import paper_style as ps
from analysis.common import derived, save_pdf

BUDGETS = [100, 125, 150, 200]


def generate(out: Path) -> None:
    d = derived("e4_low_budget_decisions")
    d = d[d["budget_mm"].isin(BUDGETS)].copy()
    d["variant"] = "max " + d["depths"].str.extract(r"(\d+)\]$")[0] + " mm, every " \
        + d["interval"].astype(str) + " d"
    variants = sorted(d["variant"].unique())
    colors = [ps.TAB10[c] for c in ("blue", "orange", "green", "red")]
    fig, axes = plt.subplots(2, len(BUDGETS), figsize=(3.2 * len(BUDGETS), 5.2), sharey="row")
    for j, budget in enumerate(BUDGETS):
        b = d[d["budget_mm"] == budget]
        for k, (variant, color) in enumerate(zip(variants, colors)):
            v = b[b["variant"] == variant]
            share = v["applied_mm"].round().value_counts(normalize=True).sort_index()
            axes[0, j].plot(share.index, share.values, marker="o", ms=3, color=color, label=variant)
            curve = v.groupby("dap")["remaining_mm"].median()
            axes[1, j].plot(curve.index, curve.values, color=color)
        axes[0, j].set(title=f"{budget} mm", xlabel="Applied depth (mm)")
        axes[1, j].set(xlabel="Days after planting")
        for ax in axes[:, j]:
            ax.grid(**ps.GRID_KW)
    axes[0, 0].set_ylabel("Share of decisions")
    axes[1, 0].set_ylabel("Remaining budget (mm)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_pdf(fig, out)
