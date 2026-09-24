"""Run one experiment config.

    python scripts/run.py config/experiments/e2_ablation.yaml --workers 60
    python scripts/run.py config/experiments/e2_ablation.yaml --resume results/runs/2026-09-24_e2_ablation
    python scripts/run.py config/experiments/e2_ablation.yaml --dry-run
"""

import os

# One thread per job. Set before numpy and torch are imported; worker processes inherit it.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
os.environ.setdefault("DEVELOPMENT", "DEVELOPMENT")

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

from ditrl.config import expand, load_yaml  # noqa: E402
from ditrl.runner import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--resume", type=Path, help="continue an existing run folder")
    ap.add_argument("--limit", type=int, help="run only the first N pending jobs")
    ap.add_argument("--tag", default="", help="suffix for the run folder name")
    ap.add_argument("--dry-run", action="store_true", help="list the jobs and stop")
    args = ap.parse_args()

    if args.dry_run:
        jobs = expand(load_yaml(args.experiment))
        for kind, n in Counter(j["kind"] for j in jobs).items():
            print(f"{n:5d} {kind} jobs")
        for j in jobs[:5]:
            print("  e.g.", j["id"])
        return
    run(args.experiment, args.workers, args.resume, args.limit, args.tag)


if __name__ == "__main__":
    main()
