"""Fig: main effects of the three DRL1 design choices on test yield (E2)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from analysis import paper_style as ps
from analysis.common import save_pdf, test_seasons
from analysis.stats import main_effects

# column -> (DRL1 level, alternative level), as stored in the derived tables
FACTORS = {"observation": ("DRL1", "KELLY"), "constraint": ("mask", "clip"),
           "reward": ("dense", "terminal")}
LABELS = {"observation": "18 instead of 5 variables", "constraint": "clip instead of mask",
          "reward": "terminal instead of stepwise reward"}


# DRL1 and the seven ablation cells (config/agents.yaml); DRL2 and the rules are left out,
# because they differ from DRL1 also in action set and decision interval.
AGENTS = ["drl1", "drl1_obs", "drl1_clip", "drl1_terminal", "drl1_obs_clip", "drl1_obs_term",
          "drl1_clip_term", "drl1_all_three"]


def effects():
    df = test_seasons("ablation")
    return main_effects(df[df["agent"].isin(AGENTS)], FACTORS)


def generate(out: Path) -> None:
    table = effects()
    sites = sorted(table["site"].unique())
    fig, axes = plt.subplots(1, len(sites), figsize=(4.2 * len(sites), 3.2), squeeze=False)
    colors = [ps.TAB10[c] for c in ("blue", "orange", "green")]
    for ax, site in zip(axes[0], sites):
        t = table[table["site"] == site]
        for i, (factor, color) in enumerate(zip(FACTORS, colors)):
            g = t[t["factor"] == factor].sort_values("budget_mm")
            ax.errorbar(g["budget_mm"] + (i - 1) * 4, g["effect_t_ha"],
                        yerr=[g["effect_t_ha"] - g["ci_low"], g["ci_high"] - g["effect_t_ha"]],
                        fmt="o", ms=4, capsize=2, color=color, label=LABELS[factor])
        ax.axhline(0, color="k", lw=0.8)
        ax.set(title=site.capitalize(), xlabel="Seasonal budget (mm)",
               ylabel="Yield change vs. DRL1 (t/ha)")
        ax.grid(**ps.GRID_KW)
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    table.to_csv(out.with_suffix(".csv"), index=False)
    save_pdf(fig, out)
