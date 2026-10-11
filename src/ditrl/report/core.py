"""What every report module shares: where it reads and writes, and what it hands back."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ditrl.plan import Plan
from ditrl.store import read_csv

ARCHIVE = Path("archive") / "fixed-300k-steps"   # first round, inside the results folder


@dataclass
class Entry:
    """One figure or table of the report."""
    path: Path            # file without suffix for figures (PNG and PDF), with suffix for tables
    title: str
    shows: str            # one or two sentences: what the reader sees
    module: str           # the generating module, e.g. ditrl.report.agreed


@dataclass
class Section:
    """What a module adds to the report: a heading and its figures and tables."""
    title: str = ""
    entries: list[Entry] = field(default_factory=list)


@dataclass
class Context:
    plan: Plan
    results: Path         # result tables (results/)
    out: Path             # report folder (reports/)
    facts: dict = field(default_factory=dict)   # tables a module hands to the README block

    @property
    def figures(self) -> Path:
        return self.out / "figures"

    @property
    def tables(self) -> Path:
        return self.out / "tables"

    def table(self, experiment: str, part: str, archive: bool = False) -> pd.DataFrame | None:
        """One part (jobs, seasons, learning, convergence, training, ...) of an experiment."""
        folder = self.results / (ARCHIVE if archive else Path()) / experiment
        for name in (f"{part}.csv", f"{part}.csv.gz"):
            if (folder / name).exists():
                return read_csv(folder / name).assign(experiment=experiment)
        return None

    def tables_of(self, experiments: list[str], part: str, archive: bool = False) -> pd.DataFrame:
        frames = [f for e in experiments if (f := self.table(e, part, archive)) is not None]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def markdown(df: pd.DataFrame, decimals: int = 2) -> str:
    """A small table as GitHub Markdown, numbers rounded."""
    shown = df.copy()
    for column in shown.select_dtypes("number"):
        shown[column] = shown[column].map(lambda v: "" if pd.isna(v) else f"{v:.{decimals}f}"
                                          if float(v) != int(v) else f"{int(v)}")
    head = "| " + " | ".join(map(str, shown.columns)) + " |"
    rule = "|" + "|".join("---" for _ in shown.columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False)]
    return "\n".join([head, rule, *rows]) + "\n"


def write_table(df: pd.DataFrame, path: Path, decimals: int = 2) -> None:
    """A table as Markdown (to read on GitHub) and CSV (to compute with)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(markdown(df, decimals), encoding="utf-8")
