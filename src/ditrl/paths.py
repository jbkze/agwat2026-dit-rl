"""Locations inside the repository."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "config" / "plan.yaml"
RESULTS = REPO / "results"        # default result root; a run may write anywhere else
LOCAL_PATHS = REPO / "config" / "paths.local.yaml"   # machine-specific, not committed
