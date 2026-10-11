"""Overview of the experiments: the two sites, the decision loop of DRL1 with the design choices
the study varies, and what DRL1 is compared with. Drawn in the visual style of the group's system
figures: black on white, sans serif, framed regions with a bold title, white rounded boxes for
steps, grey boxes for data, and a dial on every choice whose variants the study compares."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from ditrl.analysis import style
from ditrl.report.core import Context, Entry, Section

FONT = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 6, "axes.grid": False}
WIDTH, HEIGHT = 17.1, 6.4           # drawing units (about cm at the printed width)
GREY = "0.94"
FRAME = "0.45"
NOTE = "0.35"
LINE = 0.45
LINE_GAP = 0.27                     # between two text lines in a box


def _region(ax, x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.15", facecolor="none",
                                edgecolor=FRAME, linewidth=0.5))
    ax.text(x0 + 0.12, y1 - 0.12, title, ha="left", va="top", fontweight="bold", fontsize=6.5)


def _box(ax, x, y, w, h, title, lines=(), data=False) -> dict:
    """A step (white, rounded) or data (grey, square) centred at (x, y): a bold title and a few
    short lines below it."""
    shape = "square,pad=0" if data else "round,pad=0,rounding_size=0.08"
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=shape,
                                facecolor=GREY if data else "white", edgecolor="black",
                                linewidth=LINE))
    rows = [title, *lines]
    top = y + (len(rows) - 1) * LINE_GAP / 2
    for i, row in enumerate(rows):
        ax.text(x, top - i * LINE_GAP, row, ha="center", va="center",
                fontweight="bold" if i == 0 and lines else None)
    return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2)}


def _knob(ax, x, y, w, h, title, lines) -> dict:
    """A design choice the study varies: a grey box with a dial."""
    sides = _box(ax, x, y, w, h, title, lines, data=True)
    ax.add_patch(Circle((x + w / 2, y + h / 2), 0.13, facecolor="white", edgecolor="black",
                        linewidth=0.5, zorder=5))
    ax.plot([x + w / 2, x + w / 2 + 0.065], [y + h / 2, y + h / 2 + 0.11], color="black",
            linewidth=0.8, zorder=6)
    return sides


def _cloud(ax, x, y, rain: bool):
    """A small cloud, with drops when it rains and crossed out when it does not."""
    for dx, dy, r in ((-0.16, 0, 0.13), (0.0, 0.07, 0.17), (0.17, 0, 0.13)):
        ax.add_patch(Circle((x + dx, y + dy), r, facecolor="white", edgecolor="black",
                            linewidth=LINE, zorder=5))
    ax.add_patch(Rectangle((x - 0.16, y - 0.13), 0.33, 0.12, facecolor="white", edgecolor="none",
                           zorder=6))
    ax.plot([x - 0.29, x + 0.30], [y - 0.13, y - 0.13], color="black", linewidth=LINE, zorder=6)
    if rain:
        for dx in (-0.15, 0.0, 0.15):
            ax.plot([x + dx, x + dx - 0.05], [y - 0.2, y - 0.36], color="black", linewidth=0.7)
    else:
        ax.plot([x - 0.35, x + 0.35], [y - 0.3, y + 0.32], color="black", linewidth=0.8, zorder=7)


def _line(ax, *points):
    """A polyline through the points, with an arrow head at the last one."""
    xs, ys = zip(*points, strict=True)
    ax.plot(xs, ys, color="black", linewidth=LINE)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=5,
                                 color="black", linewidth=0, shrinkA=0, shrinkB=0))


def _note(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="top", fontsize=5.3, color=NOTE, style="italic")


def build(ctx: Context) -> Section:
    with plt.rc_context(FONT):
        fig, ax = plt.subplots(figsize=(7.0, 7.0 * HEIGHT / WIDTH))
        ax.set_axis_off()
        ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
        fig.subplots_adjust(0, 0, 1, 1)

        # the two sites
        _region(ax, 0, 2.1, 4.4, 6.35, "Sites: maize from 1 May")
        tunis = _box(ax, 2.35, 5.05, 3.7, 1.25, "Tunis",
                     ["no rain, sand", "train 1981-1996, test 1980", "budgets 100-800 mm"], data=True)
        nebraska = _box(ax, 2.35, 3.25, 3.7, 1.25, "Nebraska",
                        ["rain, silty clay loam", "train years 1-70, test 71-100",
                         "budgets 75-125 mm"], data=True)
        _cloud(ax, 0.85, 5.35, rain=False)
        _cloud(ax, 0.85, 3.6, rain=True)

        # the decision loop of DRL1, with the four design choices the study varies
        _region(ax, 4.8, 2.1, 17.1, 6.35, "Decision loop of DRL1")
        env = _box(ax, 6.6, 4.15, 2.0, 1.1, "AquaCrop-OSPy", ["simulates", "the season"])
        agent = _box(ax, 15.6, 4.15, 2.0, 1.1, "PPO policy", ["10 seeds"])
        obs = _knob(ax, 9.6, 5.35, 2.9, 1.0, "Observation",
                    ["5 or 18 variables", "scaled, with a forecast"])
        reward = _knob(ax, 12.85, 5.35, 2.9, 1.0, "Reward", ["relative ET + yield", "or yield only"])
        action = _knob(ax, 12.85, 2.95, 2.9, 1.0, "Dose and interval", ["0-34 mm, every 5 days", "or larger, or daily"])
        budget = _knob(ax, 9.6, 2.95, 2.9, 1.0, "Budget limit", ["mask, reject, cut", "or penalty"])
        _line(ax, env["t"], (env["t"][0], obs["l"][1]), obs["l"])
        _line(ax, obs["r"], reward["l"])
        _line(ax, reward["r"], (agent["t"][0], reward["r"][1]), agent["t"])
        _line(ax, agent["b"], (agent["b"][0], action["r"][1]), action["r"])
        _line(ax, action["l"], budget["r"])
        _line(ax, budget["l"], (env["b"][0], budget["l"][1]), env["b"])
        for site, y in ((tunis, 4.3), (nebraska, 4.0)):
            _line(ax, site["r"], (4.95, site["r"][1]), (4.95, y), (env["l"][0], y))
        _note(ax, 11.2, 4.45, "observation x budget limit x reward: 2 x 4 x 2 design")
        _note(ax, 12.85, 2.35, "larger or daily doses: Tunis 100-200 mm")

        # what DRL1 is compared with, and how
        _region(ax, 0, 0, 10.6, 1.8, "Compared with")
        _box(ax, 1.85, 0.75, 3.2, 0.75, "DRL2", ["setup of Kelly et al."])
        _box(ax, 5.3, 0.75, 3.2, 0.75, "Rules", ["threshold, fixed interval"])
        _box(ax, 8.75, 0.75, 3.2, 0.75, "Optimum", ["weather known in advance"])
        _region(ax, 11.0, 0, 17.1, 1.8, "Evaluation")
        _box(ax, 14.05, 0.8, 5.6, 1.0, "Test years", ["each variant paired with DRL1 of the same seed",
                                                 "benchmark policies also with noisy observations"])
        path = ctx.figures / "overview"
        style.save(fig, path)
    return Section("Experiments", entries=[Entry(
        path, "The experiments",
        "The two sites, the decision loop of DRL1 with the four design choices the study varies "
        "(dials), and what DRL1 is compared with.", __name__)])
