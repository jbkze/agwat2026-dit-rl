"""Statistics shared by the figure and table generators.

All DRL comparisons are paired: two variants are compared on the same site, budget, seed and
test year, so that weather and seed effects cancel out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CELL = ["site", "budget_mm", "seed", "year"]  # what a paired comparison holds fixed


def bootstrap_ci(values, n: int = 10_000, level: float = 0.95, seed: int = 0) -> tuple:
    """Percentile bootstrap interval of the mean."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values))).mean(axis=1)
    return tuple(np.quantile(means, [(1 - level) / 2, (1 + level) / 2]))


def holm(p_values) -> np.ndarray:
    """Holm-adjusted p-values (family-wise error rate)."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def main_effects(df: pd.DataFrame, factors: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Effect of switching each factor from its DRL1 level to the alternative level.

    `factors` maps a column to (DRL1 level, alternative level). For each factor, rows are paired
    on everything else (other factors and CELL). The effect is the mean yield difference
    alternative minus DRL1 per site and budget, with bootstrap interval, Wilcoxon signed-rank
    test and Holm correction over all rows of the returned table.
    """
    rows = []
    for factor, (base, alt) in factors.items():
        others = [f for f in factors if f != factor]
        wide = df.pivot_table(index=CELL + others, columns=factor, values="yield_t_ha")
        diff = (wide[alt] - wide[base]).dropna().rename("diff").reset_index()
        for (site, budget), g in diff.groupby(["site", "budget_mm"]):
            nonzero = g["diff"].abs() > 1e-9
            p = wilcoxon(g["diff"][nonzero]).pvalue if nonzero.sum() >= 5 else np.nan
            lo, hi = bootstrap_ci(g["diff"])
            rows.append({"factor": factor, "site": site, "budget_mm": budget,
                         "effect_t_ha": g["diff"].mean(), "ci_low": lo, "ci_high": hi,
                         "pairs": len(g), "p": p})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].fillna(1.0))
    return table


def lower_tail(yields, q: float = 0.1) -> dict:
    """10 % quantile and CVaR (mean of the worst 10 % of years)."""
    y = np.sort(np.asarray(yields, float))
    k = max(1, int(np.ceil(q * len(y))))
    return {"q10": float(np.quantile(y, q)), "cvar10": float(y[:k].mean())}
