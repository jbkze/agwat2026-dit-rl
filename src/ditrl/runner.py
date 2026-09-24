"""Running an experiment: run folder, manifest, parallel jobs, resume.

Layout of one run:

    results/runs/<YYYY-MM-DD>_<experiment>/
        experiment.yaml        copy of the experiment config
        config_snapshot/       copies of sites.yaml and agents.yaml at start
        manifest.json          git commit, dirty flag, package versions, machine, command
        jobs.jsonl             every job of the experiment, one JSON object per line
        run.log                progress of the whole run
        jobs/<job id>/         job.log, result.json and the files of that job

A job counts as done when its `result.json` exists. Restarting with `--resume` skips done jobs,
so an interrupted campaign continues where it stopped. A failed job writes `error.txt` and does
not stop the others. Run folders are never overwritten.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import metadata
from pathlib import Path

from ditrl.config import CONFIG, expand, load_yaml
from ditrl.sites import REPO

log = logging.getLogger("ditrl")
PACKAGES = ("aquacrop", "numpy", "pandas", "gymnasium", "stable-baselines3", "sb3-contrib",
            "torch", "pymoo")
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip()


def manifest() -> dict:
    return {
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "packages": {p: _version(p) for p in PACKAGES},
    }


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _pin_threads() -> None:
    """One job uses one core. Without this, every PyTorch process starts one thread per core
    and parallel jobs slow each other down."""
    import torch
    torch.set_num_threads(1)


def keep_awake() -> None:
    """Stop Windows from going to standby while the run is active (the check runs on
    2026-09-24 stalled for more than two hours during standby). Closing the lid still suspends."""
    if sys.platform == "win32":
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def run_job(job: dict, job_dir: str) -> tuple[str, str]:
    """Worker process: run one job and write result.json (or error.txt)."""
    from ditrl.jobs import HANDLERS

    _pin_threads()
    out = Path(job_dir)
    out.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out / "job.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    start = time.time()
    try:
        log.info("start job %s", job["id"])
        result = HANDLERS[job["kind"]](job, out)
        result["seconds"] = round(time.time() - start, 1)
        tmp = out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1, default=float))
        tmp.replace(out / "result.json")  # atomic: a half-written result never counts as done
        log.info("done in %.0f s", result["seconds"])
        return job["id"], "ok"
    except Exception:
        (out / "error.txt").write_text(traceback.format_exc())
        log.exception("job failed")
        return job["id"], "failed"


def run(experiment: Path, workers: int, resume: Path | None = None, limit: int | None = None,
        tag: str = ""):
    exp = load_yaml(experiment)
    if resume is None:
        run_dir = REPO / "results" / "runs" / f"{dt.date.today():%Y-%m-%d}_{exp['name']}{tag}"
        if run_dir.exists():
            raise FileExistsError(f"{run_dir} exists; use --resume to continue it")
        run_dir.mkdir(parents=True)
        shutil.copy(experiment, run_dir / "experiment.yaml")
        (run_dir / "config_snapshot").mkdir()
        for name in ("sites.yaml", "agents.yaml"):
            shutil.copy(CONFIG / name, run_dir / "config_snapshot" / name)
        (run_dir / "manifest.json").write_text(json.dumps(manifest(), indent=1))
        jobs = expand(exp)
        with open(run_dir / "jobs.jsonl", "w") as f:
            f.writelines(json.dumps(j) + "\n" for j in jobs)
    else:
        run_dir = resume
        jobs = [json.loads(line) for line in (run_dir / "jobs.jsonl").read_text().splitlines()]
        with open(run_dir / "manifest_resumes.jsonl", "a") as f:
            f.write(json.dumps(manifest()) + "\n")

    logging.basicConfig(level=logging.INFO, format=FORMAT, handlers=[
        logging.StreamHandler(), logging.FileHandler(run_dir / "run.log", encoding="utf-8")])
    pending = [j for j in jobs if not (run_dir / "jobs" / j["id"] / "result.json").exists()]
    n_done = len(jobs) - len(pending)
    pending = pending[:limit] if limit else pending
    log.info("%s: %d jobs, %d done, %d to run with %d workers", run_dir.name, len(jobs), n_done,
             len(pending), workers)
    if _git("status", "--porcelain"):
        log.warning("working tree has uncommitted changes; results are not tied to a commit")

    keep_awake()
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, j, str(run_dir / "jobs" / j["id"])) for j in pending]
        for i, fut in enumerate(as_completed(futures), 1):
            job_id, status = fut.result()
            failed += status == "failed"
            log.info("[%d/%d] %s %s", i, len(pending), job_id, status)
    log.info("finished: %d jobs ok, %d failed", len(pending) - failed, failed)
    return run_dir
