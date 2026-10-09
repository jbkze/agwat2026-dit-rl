"""Running jobs in parallel, one process per job, and recording how they were run.

Jobs that are already done are skipped, so calling `run` again after an interruption continues
where it stopped. A failed job writes `error.txt` and does not stop the others.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import metadata

from ditrl.paths import REPO
from ditrl.plan import Job, Plan
from ditrl.store import RESULT, Store, write_json

log = logging.getLogger("ditrl")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
PACKAGES = ("aquacrop", "numpy", "pandas", "gymnasium", "stable-baselines3", "sb3-contrib",
            "torch", "pymoo")
COMMIT_VARIABLE = "DITRL_COMMIT"   # set by the container build, which has no .git


def git(*args: str) -> str:
    """Output of a git command in the repository, or "" where there is no git."""
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    return out.stdout.strip()


def manifest(selection: list[str], shard: str | None, workers: int) -> dict:
    """Everything needed to tell how a set of results was produced."""
    return {
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "selection": selection,
        "shard": shard,
        "workers": workers,
        "commit": git("rev-parse", "HEAD") or os.environ.get(COMMIT_VARIABLE, "unknown"),
        "uncommitted_changes": bool(git("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "packages": {p: metadata.version(p) for p in PACKAGES},
    }


def _single_thread() -> None:
    """One torch thread per job: parallel jobs would otherwise compete for every core. The
    thread variables of numpy are set by the command line before anything is imported."""
    import torch
    torch.set_num_threads(1)


def _keep_awake() -> None:
    """Stop Windows from going to standby during a run (a laptop run stalled for two hours on
    2026-09-24). Closing the lid still suspends."""
    if sys.platform == "win32":
        import ctypes
        continuous, system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(continuous | system_required)


def run_job(job: Job, plan: Plan, store: Store) -> tuple[str, str]:
    """Worker process: run one job and write its result.json, or error.txt if it fails."""
    from ditrl.tasks import TASKS

    _single_thread()
    out = store.job_dir(job)
    out.mkdir(parents=True, exist_ok=True)
    (out / "error.txt").unlink(missing_ok=True)
    write_json(out / "job.json", job.to_dict())
    handler = logging.FileHandler(out / "job.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().handlers = [handler]
    logging.getLogger().setLevel(logging.INFO)
    start = time.time()
    try:
        log.info("start %s/%s", job.experiment, job.id)
        result = TASKS[job.kind](job, plan, out, store)
        result["seconds"] = round(time.time() - start, 1)
        write_json(out / RESULT, result)
        log.info("done in %.0f s", result["seconds"])
        return job.id, "ok"
    except Exception:
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log.exception("job failed")
        return job.id, "failed"


def run(jobs: list[Job], plan: Plan, store: Store, workers: int, selection: list[str],
        shard: str | None) -> int:
    """Run the jobs that are not done yet; returns the number of failed jobs."""
    store.manifests.mkdir(parents=True, exist_ok=True)
    info = manifest(selection, shard, workers)
    stamp = info["started"].replace(":", "")
    write_json(store.manifests / f"{stamp}_{info['host']}_{os.getpid()}.json", info)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    pending = [job for job in jobs if not store.done(job)]
    log.info("%d jobs, %d done, %d to run with %d workers", len(jobs), len(jobs) - len(pending),
             len(pending), workers)
    if info["uncommitted_changes"]:
        log.warning("uncommitted changes: these results are not tied to a commit")
    _keep_awake()
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, job, plan, store) for job in pending]
        for i, future in enumerate(as_completed(futures), 1):
            job_id, status = future.result()
            failed += status == "failed"
            log.info("[%d/%d] %s %s", i, len(pending), job_id, status)
    log.info("finished: %d ok, %d failed", len(pending) - failed, failed)
    return failed
