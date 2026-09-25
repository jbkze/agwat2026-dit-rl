"""Small helpers for the figure and table generators."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.collect import DERIVED, REPO, analysis_config

METHOD_ORDER = ["drl1", "drl2", "threshold", "fixed_interval", "full_irrigation"]
METHOD_LABEL = {"drl1": "DRL1", "drl2": "DRL2", "threshold": "Threshold rule",
                "fixed_interval": "Fixed interval", "full_irrigation": "Full irrigation",
                "drl1_forecast": "DRL1 + 7-day forecast"}


def derived(name: str) -> pd.DataFrame:
    """A table from results/derived/. Fails with a hint if `make collect` has not run yet."""
    path = DERIVED / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the stage that produces it, then make collect")
    return pd.read_csv(path)


def test_seasons(name: str) -> pd.DataFrame:
    df = derived(f"{name}_seasons")
    return df[df["split"] == "test"]


def run_dir(name: str) -> Path:
    """First existing run folder of a derived table in config/analysis.yaml."""
    runs = analysis_config()["runs"][name]
    if not runs:
        raise FileNotFoundError(f"no run for {name!r} yet (config/analysis.yaml)")
    return REPO / runs[0]


def save_pdf(fig, out: Path) -> None:
    """Deterministic PDF (no creation date), so that the drift check can compare hashes."""
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
