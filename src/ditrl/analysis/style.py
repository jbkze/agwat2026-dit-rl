"""Figure style for every figure of the repository: SciencePlots (`science` without LaTeX), Paul
Tol's colour-blind safe "bright" palette, fixed names and colours per method, and writers that
produce files without a creation date, so that unchanged figures stay unchanged in git."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: E402, F401  (registers the styles)

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
                     "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linestyle": "--", "grid.linewidth": 0.5})
# The AquaCrop fork sets the root logger to INFO, which would print every font subsetting step.
logging.getLogger("fontTools").setLevel(logging.WARNING)

BRIGHT = {"blue": "#4477AA", "cyan": "#66CCEE", "green": "#228833", "yellow": "#CCBB44",
          "red": "#EE6677", "purple": "#AA3377", "grey": "#BBBBBB"}
TAB10 = BRIGHT   # name kept for the generators that ask for a palette by colour name
GRID = dict(linestyle="--", linewidth=0.5, color="gray", alpha=0.3)
BAND_ALPHA = 0.2
PANEL_WIDTH = 3.3    # inches per panel
PANEL_HEIGHT = 2.6
SMALL_PANEL = 2.4
TALL_PANEL = (3.0, 4.6)
DOT_ALPHA = 0.7
MEAN_BAR_HALF_WIDTH = 0.25   # in units of a categorical x axis
TICK_ROTATION = 45

METHODS = {   # name -> (label, colour), in the order of the legends
    "drl1": ("DRL1", BRIGHT["blue"]),
    "drl2": ("DRL2", BRIGHT["red"]),
    "threshold": ("Threshold rule", BRIGHT["green"]),
    "fixed_interval": ("Fixed interval", BRIGHT["purple"]),
    "full_irrigation": ("Full irrigation", BRIGHT["grey"]),
    "drl1_forecast": ("DRL1 + 7-day forecast", BRIGHT["cyan"]),
}
# Colours of the variants agreed for the ablation, in plotting order.
AGREED = {"drl1": BRIGHT["blue"], "drl1_obs": BRIGHT["green"], "drl1_reject": BRIGHT["red"],
          "drl1_terminal": BRIGHT["yellow"]}


def label(method: str) -> str:
    return METHODS.get(method, (method, None))[0]


def save_pdf(fig, out: Path) -> None:
    """PDF without a creation date, so that the drift check can compare file hashes."""
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def save(fig, stem: Path) -> None:
    """The figure as PNG (for the README) and PDF (for the paper), both without dates."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", metadata={"Software": None})
    save_pdf(fig, stem.with_suffix(".pdf"))
