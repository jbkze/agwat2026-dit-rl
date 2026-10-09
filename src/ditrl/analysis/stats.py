"""Statistics shared by the figures and tables.

DRL comparisons are paired: two methods are compared on the same site, budget, seed and test
year, so that weather and seed effects cancel out.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

POLICY = ["method", "site", "budget_mm", "seed"]       # one trained policy
PAIR = ["site", "budget_mm", "seed", "year"]          # what a paired comparison holds fixed
NONZERO_MM = 1e-9                                     # differences below this count as ties


def lower_quantile(values, q: float) -> float:
    """The q quantile as an order statistic: the k-th lowest value with k = floor(q n) + 1, so
    the fourth-lowest of 30 years for q = 0.1. No interpolation between years."""
    ordered = np.sort(np.asarray(values, float))
    return float(ordered[min(math.floor(q * len(ordered)), len(ordered) - 1)])


def bootstrap_ci(values, resamples: int, level: float, seed: int) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean."""
    values = np.asarray(values, float)
    means = np.random.default_rng(seed).choice(values, size=(resamples, len(values))).mean(axis=1)
    return tuple(np.quantile(means, [(1 - level) / 2, (1 + level) / 2]))


def holm(p_values) -> np.ndarray:
    """Holm-adjusted p-values (family-wise error rate)."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def per_policy(seasons: pd.DataFrame, q: float) -> pd.DataFrame:
    """Mean and lower quantile of test yield for every trained policy (years are not pooled
    across seeds)."""
    return (seasons.groupby(POLICY)["yield_t_ha"]
            .agg(mean_yield="mean", low_yield=lambda y: lower_quantile(y, q), years="size")
            .reset_index())


def paired_differences(seasons: pd.DataFrame, reference: str, q: float) -> pd.DataFrame:
    """Per method, site, budget and seed: reference minus method, for the mean and the lower
    quantile of test yield. Positive values mean that the change away from the reference costs
    yield."""
    policies = per_policy(seasons, q)
    ref = policies[policies["method"] == reference].drop(columns=["method", "years"])
    other = policies[policies["method"] != reference]
    pairs = other.merge(ref, on=["site", "budget_mm", "seed"], suffixes=("", "_reference"))
    pairs["mean_difference"] = pairs["mean_yield_reference"] - pairs["mean_yield"]
    pairs["low_difference"] = pairs["low_yield_reference"] - pairs["low_yield"]
    return pairs


def main_effects(seasons: pd.DataFrame, factors: dict[str, list[str]], reference: dict[str, str],
                 bootstrap: dict, min_pairs: int) -> pd.DataFrame:
    """Effect of switching one factor from its reference level to each other level, paired on
    all other factors and on site, budget, seed and year: mean difference (other level minus
    reference level) per site and budget, with bootstrap interval, Wilcoxon signed-rank test and
    Holm correction over all rows of the returned table."""
    rows = []
    for factor, levels in factors.items():
        others = [f for f in factors if f != factor]
        wide = seasons.pivot_table(index=PAIR + others, columns=factor, values="yield_t_ha")
        for level in levels:
            if level == reference[factor] or level not in wide or reference[factor] not in wide:
                continue
            diff = (wide[level] - wide[reference[factor]]).dropna().rename("diff").reset_index()
            for (site, budget), g in diff.groupby(["site", "budget_mm"]):
                nonzero = g["diff"].abs() > NONZERO_MM
                p = wilcoxon(g["diff"][nonzero]).pvalue if nonzero.sum() >= min_pairs else np.nan
                low, high = bootstrap_ci(g["diff"], **bootstrap)
                rows.append({"factor": factor, "level": level, "site": site, "budget_mm": budget,
                             "effect_t_ha": g["diff"].mean(), "ci_low": low, "ci_high": high,
                             "pairs": len(g), "p": p})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm(table["p"].fillna(1.0))
    return table
