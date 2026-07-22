"""Event study: test the management_qa_divergence feature directly against
forward abnormal returns, before any predictive model (Phase 11).

Operates on a table with one row per call that has both a divergence
column (e.g. management_qa_divergence, from
`earnings_nlp.features.change_features`) and abnormal return columns
(e.g. abnormal_return_1d/5d/20d, from
`earnings_nlp.backtest.event_returns`).
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats.mstats import winsorize

QUINTILE_COLOR = "#2E5FA3"  # single-hue base for sequential quintile shading


def assign_divergence_quintiles(
    df: pd.DataFrame,
    divergence_col: str = "management_qa_divergence",
    n_quantiles: int = 5,
    quintile_col: str = "divergence_quintile",
) -> pd.DataFrame:
    """Assign each call to a divergence quintile: 1 = lowest divergence,
    `n_quantiles` = highest. Raises ValueError rather than silently
    forming fewer, larger groups when there isn't enough data -- calling
    a 2-group split "quintiles" would misrepresent how granular the
    result actually is. Warns (but proceeds) if duplicate divergence
    values force fewer groups than requested.
    """
    valid = df[divergence_col].notna()
    n = int(valid.sum())
    if n < n_quantiles:
        raise ValueError(
            f"cannot form {n_quantiles} quintiles from {n} calls with a "
            f"non-null {divergence_col}; need at least {n_quantiles} calls, "
            f"and ideally many more per bucket for a meaningful result"
        )

    out = df.copy()
    out[quintile_col] = pd.NA
    out.loc[valid, quintile_col] = (
        pd.qcut(out.loc[valid, divergence_col], n_quantiles, labels=False, duplicates="drop") + 1
    )

    actual_groups = out[quintile_col].nunique()
    if actual_groups < n_quantiles:
        warnings.warn(
            f"requested {n_quantiles} quintiles but {divergence_col} has too "
            f"many tied/duplicate values to support that; formed {actual_groups} "
            f"groups instead. Results below reflect {actual_groups} groups, not "
            f"true quintiles."
        )
    return out


def quintile_return_summary(
    df: pd.DataFrame,
    return_cols: list[str],
    quintile_col: str = "divergence_quintile",
) -> pd.DataFrame:
    """Mean, median, and count of each return column, grouped by quintile."""
    grouped = df.groupby(quintile_col)[return_cols]
    summary = grouped.agg(["mean", "median", "count"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    return summary.reset_index()


def is_monotonic(
    summary_df: pd.DataFrame,
    return_mean_col: str,
    quintile_col: str = "divergence_quintile",
    direction: str = "decreasing",
) -> bool:
    """Whether mean returns move monotonically across quintiles 1..N, in
    the given `direction` ("decreasing" or "increasing").
    """
    ordered = summary_df.sort_values(quintile_col)[return_mean_col]
    if direction == "decreasing":
        return bool((ordered.diff().dropna() <= 0).all())
    if direction == "increasing":
        return bool((ordered.diff().dropna() >= 0).all())
    raise ValueError(f"unknown direction: {direction!r}")


def long_short_spread(
    summary_df: pd.DataFrame,
    return_mean_col: str,
    quintile_col: str = "divergence_quintile",
) -> dict:
    """Long-low/short-high divergence spread: mean return of the lowest
    quintile minus mean return of the highest quintile. A positive spread
    supports the hypothesis that high management_qa_divergence predicts
    underperformance.
    """
    low_group = summary_df[quintile_col].min()
    high_group = summary_df[quintile_col].max()
    low_mean = summary_df.loc[summary_df[quintile_col] == low_group, return_mean_col].iloc[0]
    high_mean = summary_df.loc[summary_df[quintile_col] == high_group, return_mean_col].iloc[0]
    return {
        "low_quintile": low_group,
        "high_quintile": high_group,
        "low_mean_return": low_mean,
        "high_mean_return": high_mean,
        "long_short_spread": low_mean - high_mean,
    }


def winsorize_returns(df: pd.DataFrame, return_col: str, limits: tuple = (0.01, 0.01)) -> pd.DataFrame:
    """Return a copy of `df` with `return_col` winsorized at `limits`
    (fraction trimmed from the bottom, top), for the "does it remain after
    removing extreme returns?" robustness check.
    """
    out = df.copy()
    values = out[return_col].to_numpy(dtype=float)
    mask = ~pd.isna(values)
    winsorized = values.copy()
    winsorized[mask] = winsorize(values[mask], limits=limits)
    out[return_col] = winsorized
    return out


def quintile_summary_by_group(
    df: pd.DataFrame,
    group_col: str,
    return_col: str,
    quintile_col: str = "divergence_quintile",
) -> pd.DataFrame:
    """Mean/median/count of `return_col` by (group_col, quintile_col), for
    "is it concentrated in certain sectors / years?" breakdowns. `group_col`
    is generic -- pass a year or sector column.
    """
    grouped = df.groupby([group_col, quintile_col])[return_col]
    summary = grouped.agg(["mean", "median", "count"]).reset_index()
    return summary.rename(columns={"mean": f"{return_col}_mean", "median": f"{return_col}_median", "count": f"{return_col}_count"})


def conditional_long_short_spread(
    df: pd.DataFrame,
    condition_mask: pd.Series,
    return_col: str,
    divergence_col: str = "management_qa_divergence",
    n_quantiles: int = 5,
) -> dict:
    """Long-short spread computed separately within condition_mask=True and
    condition_mask=False subsets, for "is it stronger when analyst
    questions are negative?"-type comparisons. Raises the same ValueError
    as `assign_divergence_quintiles` if either subset is too small.
    """
    results = {}
    for label, subset in [("condition_true", df[condition_mask]), ("condition_false", df[~condition_mask])]:
        quintiled = assign_divergence_quintiles(subset, divergence_col, n_quantiles)
        summary = quintile_return_summary(quintiled, [return_col])
        results[label] = long_short_spread(summary, f"{return_col}_mean")
    return results


def plot_divergence_scatter(
    df: pd.DataFrame,
    divergence_col: str,
    return_col: str,
    save_path=None,
):
    """Scatter of divergence (x) vs. forward abnormal return (y) for every
    call -- the rawest possible view of the relationship, before any
    quintile binning.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df[divergence_col], df[return_col], s=36, color=QUINTILE_COLOR, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#999999", linewidth=1, zorder=0)
    ax.axvline(0, color="#999999", linewidth=1, zorder=0)
    ax.set_xlabel(divergence_col)
    ax.set_ylabel(return_col)
    ax.set_title(f"{return_col} vs. {divergence_col}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_divergence_distribution(df: pd.DataFrame, divergence_col: str, save_path=None):
    """Histogram of the divergence feature's distribution across calls."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(df[divergence_col].dropna(), bins=min(20, max(5, df[divergence_col].nunique())), color=QUINTILE_COLOR, edgecolor="white")
    ax.set_xlabel(divergence_col)
    ax.set_ylabel("number of calls")
    ax.set_title(f"Distribution of {divergence_col}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_quintile_mean_returns(summary_df: pd.DataFrame, return_mean_col: str, quintile_col: str = "divergence_quintile", save_path=None):
    """Bar chart of mean return by quintile, shaded light-to-dark in
    quintile order (a sequential, single-hue encoding of the ordinal
    divergence rank, not a categorical palette).
    """
    ordered = summary_df.sort_values(quintile_col)
    n = len(ordered)
    shades = [_lighten(QUINTILE_COLOR, 0.65 - 0.5 * i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(ordered[quintile_col].astype(str), ordered[return_mean_col], color=shades)
    ax.axhline(0, color="#999999", linewidth=1)
    ax.set_xlabel("divergence quintile (1 = lowest, higher = highest)")
    ax.set_ylabel(return_mean_col)
    ax.set_title(f"Mean {return_mean_col} by divergence quintile")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def _lighten(hex_color: str, amount: float) -> str:
    """Blend `hex_color` toward white by `amount` (0 = unchanged, 1 = white)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"
