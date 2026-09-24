"""Run every experiment of the plan in order, then build the derived tables and stage the paper
assets in paper_assets/ (the paper repo is not touched).

    python scripts/run_all.py --workers 14

Each experiment continues its newest run folder if one exists, so calling this script again
after an interruption resumes where it stopped. Order: the results needed first come first;
E5 needs the trained models of E1 and E2.
"""

import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
os.environ.setdefault("DEVELOPMENT", "DEVELOPMENT")

import argparse  # noqa: E402
import logging  # noqa: E402

from ditrl.config import CONFIG, latest_run  # noqa: E402
from ditrl.runner import run  # noqa: E402

ORDER = ["e1_benchmark", "e2_ablation", "e0_oracle_ds", "e4_low_budget", "e3a_forecast",
         "e3b_test_year", "e0_oracle_ss", "e5_robustness"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--only", nargs="+", choices=ORDER, help="run only these experiments")
    args = ap.parse_args()
    for name in args.only or ORDER:
        run(CONFIG / "experiments" / f"{name}.yaml", args.workers, resume=latest_run(name))
        logging.getLogger().handlers.clear()  # the next run opens its own run.log

    from analysis import assets, collect
    collect.main()
    assets.stage()  # the paper repo is never touched here; copy with `make paper-assets`


if __name__ == "__main__":
    main()
