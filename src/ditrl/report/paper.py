"""The paper figures (src/ditrl/analysis/figures.py), shown here as soon as their data exist."""

from __future__ import annotations

from ditrl.analysis import figures
from ditrl.report.core import Context, Entry, Section

FIGURES = {   # name in reports/figures -> (generator, title, what it shows)
    "paper_cwpf": (figures.cwpf, "Crop water production function",
                   "Mean test yield over the seasonal budget for every method; DRL lines show the "
                   "median over seeds, the band the worst to the best seed; dashed: the oracle."),
    "paper_actions_low_budget": (figures.actions_low_budget, "Actions at low budgets",
                                 "Which depths DRL1 and its low-budget variants apply in Tunis at "
                                 "100 to 200 mm, and how the remaining budget falls."),
    "paper_learning_curves": (figures.learning_curves, "Benchmark learning curves",
                              "Validation yield of DRL1 and DRL2 over the training steps."),
}


def build(ctx: Context) -> Section:
    section = Section("Paper figures")
    for name, (generate, title, shows) in FIGURES.items():
        path = ctx.figures / name
        try:
            generate(ctx.plan, ctx.results, path.with_suffix(".pdf"))
        except FileNotFoundError:
            continue   # its experiments have no results yet
        section.entries.append(Entry(path, title, shows, f"{figures.__name__}.{generate.__name__}"))
    return section
