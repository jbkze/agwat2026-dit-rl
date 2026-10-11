"""System overview: how a plan becomes results, figures and this report."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ditrl.analysis import style
from ditrl.report.core import Context, Entry, Section

# (x, y) centre, title, detail; drawn left to right, top row first
BOXES = [
    ((0.11, 0.70), "config/plan.yaml", "sites, methods,\nexperiments by stage"),
    ((0.37, 0.70), "ditrl run", "any machine, any shard;\ntrains until convergence"),
    ((0.63, 0.70), "runs/", "one folder per job:\nmodel, logs, curves"),
    ((0.89, 0.70), "ditrl export", "tables per experiment"),
    ((0.63, 0.22), "results/", "tables in git"),
    ((0.37, 0.22), "ditrl report", "figures, tables,\nREADME block"),
    ((0.11, 0.22), "reports/", "what you read here"),
]
ARROWS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
WIDTH, HEIGHT = 0.20, 0.30
COLORS = [style.BRIGHT["grey"], style.BRIGHT["blue"], style.BRIGHT["grey"], style.BRIGHT["blue"],
          style.BRIGHT["grey"], style.BRIGHT["blue"], style.BRIGHT["grey"]]


def build(ctx: Context) -> Section:
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    for ((x, y), title, detail), color in zip(BOXES, COLORS, strict=True):
        ax.add_patch(FancyBboxPatch((x - WIDTH / 2, y - HEIGHT / 2), WIDTH, HEIGHT,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor=color, alpha=0.25, edgecolor=color))
        ax.text(x, y + 0.06, title, ha="center", va="center", fontsize=8, fontweight="bold",
                family="monospace" if "/" in title or title.startswith("ditrl") else None)
        ax.text(x, y - 0.06, detail, ha="center", va="center", fontsize=6.5)
    for a, b in ARROWS:
        (xa, ya), (xb, yb) = BOXES[a][0], BOXES[b][0]
        if ya == yb:   # along a row
            sign = 1 if xb > xa else -1
            start, end = (xa + sign * WIDTH / 2, ya), (xb - sign * WIDTH / 2, yb)
        else:          # down to the next row
            start, end = (xa, ya - HEIGHT / 2), (xb + WIDTH / 2, yb)
        bend = "arc3,rad=0" if ya == yb else "angle,angleA=-90,angleB=0"
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                     color="0.35", linewidth=0.8, connectionstyle=bend))
    path = ctx.figures / "overview"
    style.save(fig, path)
    return Section("System", entries=[Entry(path, "System overview",
                                  "How an experiment plan becomes runs, result tables and this "
                                  "report.", __name__)])
