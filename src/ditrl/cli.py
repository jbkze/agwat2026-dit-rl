"""Command line entry point.

    ditrl list    stage1                       jobs of a selection, and how many are done
    ditrl run     stage1 --workers 8           run them; resumable
    ditrl run     stage1 --shard 2/3           one of three equal parts, e.g. one per machine
    ditrl export  stage1                       result tables of the finished jobs -> results/
    ditrl assets  stage | export | check       paper figures, tables and macros from results/

A selection is any mix of stages ("stage1") and experiment names from config/plan.yaml.
`--runs` sets the folder of the raw job results (default runs/); the runs of several machines
merge by copying their jobs/ and manifests/ folders into one.
"""

import os

# One math thread per process, set before numpy and torch are imported; workers inherit it.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

from ditrl.paths import PLAN, RESULTS, RUNS  # noqa: E402
from ditrl.plan import Plan, shard  # noqa: E402
from ditrl.store import Store  # noqa: E402

RESERVED_CORES = 2   # default workers: all cores but these, for the system and the main process


def _shard_arg(text: str) -> tuple[int, int]:
    index, count = (int(part) for part in text.split("/"))
    return index, count


def _selected_jobs(args, plan: Plan):
    jobs = plan.jobs(args.selection)
    return shard(jobs, *args.shard) if args.shard else jobs


def cmd_list(args) -> None:
    plan, store = Plan.load(args.plan), Store(args.runs)
    jobs = _selected_jobs(args, plan)
    done = Counter(j.experiment for j in jobs if store.done(j))
    for experiment, total in Counter(j.experiment for j in jobs).items():
        kinds = Counter(j.kind for j in jobs if j.experiment == experiment)
        detail = ", ".join(f"{n} {kind}" for kind, n in kinds.items())
        print(f"{experiment:26s} {total:5d} jobs ({detail}), {done[experiment]} done")
    print(f"{'total':26s} {len(jobs):5d} jobs, {sum(done.values())} done")


def cmd_run(args) -> None:
    from ditrl.runner import run
    plan = Plan.load(args.plan)
    shard_text = "/".join(map(str, args.shard)) if args.shard else None
    failed = run(_selected_jobs(args, plan), plan, Store(args.runs), args.workers,
                 args.selection, shard_text)
    raise SystemExit(1 if failed else 0)


def cmd_export(args) -> None:
    from ditrl.export import export
    plan = Plan.load(args.plan)
    export(plan, Store(args.runs), plan.select(args.selection), args.results)


def cmd_assets(args) -> None:
    from ditrl.analysis import assets
    getattr(assets, args.action)(Plan.load(args.plan), args.results)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ditrl", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=PLAN, help="experiment plan (YAML)")
    parser.add_argument("--runs", type=Path, default=RUNS, help="folder of the raw job results")
    parser.add_argument("--results", type=Path, default=RESULTS, help="folder of the result tables")
    commands = parser.add_subparsers(dest="command", required=True)

    for name, func, text in (("list", cmd_list, "show the jobs of a selection"),
                             ("run", cmd_run, "run the jobs of a selection"),
                             ("export", cmd_export, "write the result tables of a selection")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("selection", nargs="+", help="stages (stage1) and/or experiment names")
        sub.set_defaults(func=func)
        if name != "export":
            sub.add_argument("--shard", type=_shard_arg, metavar="I/N",
                             help="only part I of N equal parts of the job list")
        if name == "run":
            sub.add_argument("--workers", type=int,
                             default=max(1, (os.cpu_count() or 1) - RESERVED_CORES))

    assets = commands.add_parser("assets", help="paper figures, tables and macros")
    assets.add_argument("action", choices=["stage", "export", "check"])
    assets.set_defaults(func=cmd_assets)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
