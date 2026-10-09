"""Where raw job results live, and how they are written.

    <runs>/jobs/<experiment>/<job id>/   one folder per job, written only by that job
    <runs>/manifests/                    one file per call of `ditrl run`: code, packages, machine

A job is done when its `result.json` exists; it is written last and atomically, so a half-finished
job never counts as done and is simply run again. Because jobs never share a folder, the runs of
several machines merge by copying their `jobs/` and `manifests/` folders into one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ditrl.plan import Job

RESULT = "result.json"


class Store:
    def __init__(self, root: Path, previous: Path | None = None):
        self.root = Path(root)
        self.jobs = self.root / "jobs"
        self.manifests = self.root / "manifests"
        # Runs of an earlier round whose trained models are continued instead of started anew.
        self.previous = Store(previous) if previous else None

    def job_dir(self, job: Job | str, job_id: str | None = None) -> Path:
        """Folder of a job, given as a Job or as experiment name and job id."""
        if isinstance(job, Job):
            return self.jobs / job.experiment / job.id
        return self.jobs / job / job_id

    def resumable(self, job: Job) -> Path | None:
        """Folder of this job's trained model from the earlier round, if there is one."""
        if self.previous is None:
            return None
        folder = self.previous.job_dir(job)
        return folder if (folder / "model.zip").exists() else None

    def done(self, job: Job) -> bool:
        return (self.job_dir(job) / RESULT).exists()

    def finished(self, experiment: str):
        """Folders of the finished jobs of an experiment, sorted by job id."""
        root = self.jobs / experiment
        folders = sorted(root.iterdir()) if root.exists() else []
        return [folder for folder in folders if (folder / RESULT).exists()]


def write_json(path: Path, obj) -> None:
    """Write via a temporary file, so that a reader never sees a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    """pandas writes floats with all their digits; this parser reads them back unchanged."""
    return pd.read_csv(path, float_precision="round_trip")
