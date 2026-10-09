"""Locations inside the repository."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "config" / "plan.yaml"
RUNS = REPO / "runs"          # raw job folders (models, logs); not in git, any other root works too
RESULTS = REPO / "results"    # result tables per experiment, exported from the runs; in git
LOCAL_PATHS = REPO / "config" / "paths.local.yaml"   # machine-specific, not committed
