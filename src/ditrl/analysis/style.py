"""Figure style: embedded TrueType fonts, font sizes, a light dashed grid, the Tableau 10 colours,
fixed method names and colours, and a deterministic PDF writer."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# The AquaCrop fork sets the root logger to INFO, which would print every font subsetting step.
logging.getLogger("fontTools").setLevel(logging.WARNING)
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150,
})

GRID = dict(linestyle="--", linewidth=0.5, color="gray", alpha=0.3)
BAND_ALPHA = 0.2
PANEL_WIDTH = 4.2   # inches per site panel
PANEL_HEIGHT = 3.4
SMALL_PANEL = 2.8   # inches, width and height of one panel in a row of many
TALL_PANEL = (3.2, 5.2)   # inches, width and height of a column with two stacked panels
DOT_ALPHA = 0.7
MEAN_BAR_HALF_WIDTH = 0.25   # in units of the categorical x axis
TICK_ROTATION = 60

TAB10 = {"blue": "#4e79a7", "orange": "#f28e2b", "red": "#e15759", "teal": "#76b7b2",
         "green": "#59a14f", "purple": "#b07aa1", "brown": "#9c755f", "pink": "#ff9da7",
         "yellow": "#edc948", "gray": "#bab0ac"}

METHODS = {   # name -> (label, colour), in the order of the legends
    "drl1": ("DRL1", TAB10["blue"]),
    "drl2": ("DRL2", TAB10["orange"]),
    "threshold": ("Threshold rule", TAB10["green"]),
    "fixed_interval": ("Fixed interval", TAB10["purple"]),
    "full_irrigation": ("Full irrigation", TAB10["gray"]),
    "drl1_forecast": ("DRL1 + 7-day forecast", TAB10["teal"]),
}


def label(method: str) -> str:
    return METHODS.get(method, (method, None))[0]


def save_pdf(fig, out: Path) -> None:
    """PDF without a creation date, so that the drift check can compare file hashes."""
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
