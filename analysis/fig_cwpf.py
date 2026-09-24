"""Fig: crop water production function per site (E1). Mean yield over the test years;
DRL lines show the median over seeds, the band spans the worst to the best seed."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis import paper_style as ps
from analysis.collect import REPO, analysis_config
from analysis.common import METHOD_LABEL, METHOD_ORDER, derived, save_pdf, test_seasons

EXTERNAL = REPO / analysis_config()["external_heuristics"]
COLORS = dict(zip(METHOD_ORDER, [ps.TAB10[c] for c in ("blue", "orange", "green", "purple", "gray")]))


def generate(out: Path) -> None:
    df = test_seasons("e1_benchmark")
    per_seed = (df.groupby(["site", "agent", "budget_mm", "seed"], dropna=False)["yield_t_ha"]
                .mean().reset_index())
    oracle = derived("oracle_ds").groupby(["site", "budget_mm"])["yield_t_ha"].mean().reset_index()
    sites = sorted(df["site"].unique())
    fig, axes = plt.subplots(1, len(sites), figsize=(4.2 * len(sites), 3.4), squeeze=False)
    for ax, site in zip(axes[0], sites):
        o = oracle[oracle["site"] == site]
        ax.plot(o["budget_mm"], o["yield_t_ha"], "k--", lw=1.2, label="EA-DS (perfect foresight)")
        for agent in METHOD_ORDER:
            g = per_seed[(per_seed["site"] == site) & (per_seed["agent"] == agent)]
            if g.empty:
                continue
            if agent == "full_irrigation":
                ax.axhline(g["yield_t_ha"].iloc[0], color=COLORS[agent], lw=1, ls=":",
                           label=METHOD_LABEL[agent])
                continue
            s = g.groupby("budget_mm")["yield_t_ha"].agg(["median", "min", "max"]).reset_index()
            ax.plot(s["budget_mm"], s["median"], marker="o", ms=3, color=COLORS[agent],
                    label=METHOD_LABEL[agent])
            ax.fill_between(s["budget_mm"], s["min"], s["max"], color=COLORS[agent], alpha=0.2, lw=0)
        if EXTERNAL.exists():  # Simplex heuristics by Niels, if delivered
            ext = pd.read_csv(EXTERNAL)
            ext = ext[ext["site"] == site].groupby(["method", "budget_mm"])["yield_t_ha"].mean()
            for method, s in ext.groupby(level=0):
                ax.plot(s.index.get_level_values(1), s.values, ls="-.", lw=1, label=method)
        ax.set(title=site.capitalize(), xlabel="Seasonal budget (mm)", ylabel="Yield (t/ha)")
        ax.grid(**ps.GRID_KW)
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    per_seed.to_csv(out.with_suffix(".csv"), index=False)
    save_pdf(fig, out)
