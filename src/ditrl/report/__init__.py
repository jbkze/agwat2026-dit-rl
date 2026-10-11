"""The report of this repository, rebuilt from the result tables after every new batch of results.

    ditrl report      figures and tables in reports/, the list reports/README.md, and the
                      results block of the main README.md (status table and first figure)

One module per topic; each returns a section with its figures and tables. A module whose
experiments have no results yet adds nothing.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

from ditrl.paths import REPO
from ditrl.plan import Plan
from ditrl.report import agreed, overview, paper, progress, sensitivity
from ditrl.report.core import Context, Section, markdown

MODULES = [overview, progress, sensitivity, agreed, paper]   # order of the list
REPORTS = REPO / "reports"
README = REPO / "README.md"
START, END = "<!-- report:start -->", "<!-- report:end -->"


def _relative(path: Path, start: Path) -> str:
    return Path(path).relative_to(start).as_posix()


def _index(sections: list[Section], out: Path) -> str:
    """reports/README.md: one row per figure or table."""
    lines = ["# Report", "",
             "Built by `ditrl report` from `results/`. Figures as PNG and PDF, tables as "
             "Markdown and CSV.", "", "| File | Shows |", "|---|---|"]
    for entry in (e for s in sections for e in s.entries):
        file = entry.path.with_suffix(".png") if entry.path.suffix == "" else entry.path
        name = _relative(file, out)
        lines.append(f"| [{name}]({name}) | {entry.shows} |")
    return "\n".join(lines) + "\n"


def _readme_block(sections: list[Section], ctx: Context) -> str:
    entries = {e.path.name: e for s in sections for e in s.entries}
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [START, "## Status", "",
             f"Filled in by `ditrl report` after every batch of results (last on {stamp}).", "",
             markdown(ctx.facts["status"]).rstrip(), ""]
    if "effects_over_budget" in entries:
        e = entries["effects_over_budget"]
        lines += ["## First look", "", e.shows, "",
                  f"![{e.title}]({_relative(e.path.with_suffix('.png'), REPO)})", ""]
    lines.append(END)
    return "\n".join(lines)


def build(plan: Plan, results: Path, out: Path = REPORTS, readme: Path = README) -> None:
    for folder in ("figures", "tables"):   # nothing stale survives; unchanged files stay identical
        shutil.rmtree(out / folder, ignore_errors=True)
    ctx = Context(plan, results, out)
    sections = [module.build(ctx) for module in MODULES]
    (out / "README.md").write_text(_index(sections, out), encoding="utf-8")
    text = readme.read_text(encoding="utf-8")
    if START in text and END in text:
        head, rest = text.split(START, 1)
        tail = rest.split(END, 1)[1]
        readme.write_text(head + _readme_block(sections, ctx) + tail, encoding="utf-8")
    count = sum(len(s.entries) for s in sections)
    print(f"report: {count} figures and tables in {out}")
