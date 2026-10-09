"""Run the experiments of one or more stages of docs/experiment-plan.md, then build the derived
tables and stage the paper assets in paper_assets/ (the paper repo is not touched).

    python scripts/run_all.py --stage 1 --workers 14      # minimum
    python scripts/run_all.py --stage 2                   # everything of stages 1 and 2

A stage includes all lower stages. Each experiment continues its newest run folder if one
exists, so calling this script again after an interruption resumes where it stopped, and
calling it with a higher stage adds only the new experiments.
"""

import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
os.environ.setdefault("DEVELOPMENT", "DEVELOPMENT")

import argparse  # noqa: E402
import logging  # noqa: E402

from ditrl.config import CONFIG, latest_run  # noqa: E402
from ditrl.runner import run  # noqa: E402

STAGES = {
    # The ablation grows in steps a1 (required) to a4 (complete), see E2 of the plan.
    1: ["s1_benchmark", "a1_ablation"],
    2: ["a2_budget_handling", "a3_factorial", "s2_benchmark_seeds", "s2_benchmark_budgets",
        "s2_low_budget", "s2_robustness", "a4_seeds", "a4_budgets"],
    3: ["s3_forecast", "s3_test_year", "s3_obs_scaled", "s3_oracle_ds", "s3_oracle_ss"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--stage", type=int, choices=sorted(STAGES), default=1)
    args = ap.parse_args()
    for name in [n for s in sorted(STAGES) if s <= args.stage for n in STAGES[s]]:
        run(CONFIG / "experiments" / f"{name}.yaml", args.workers, resume=latest_run(name))
        logging.getLogger().handlers.clear()  # the next run opens its own run.log

    from analysis import assets, collect
    collect.main()
    assets.stage()  # the paper repo is never touched here; copy with `make paper-assets`


if __name__ == "__main__":
    main()
