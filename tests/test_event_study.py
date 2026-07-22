import matplotlib

matplotlib.use("Agg")  # headless: no display needed to run these tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from earnings_nlp.backtest.event_study import (
    assign_divergence_quintiles,
    conditional_long_short_spread,
    is_monotonic,
    long_short_spread,
    plot_divergence_distribution,
    plot_divergence_scatter,
    plot_quintile_mean_returns,
    quintile_return_summary,
    quintile_summary_by_group,
    winsorize_returns,
)


def _synthetic_calls(n=30, relationship="decreasing"):
    """n calls with evenly spaced, unique divergence values (no ties, so
    quintile assignment is unambiguous) and an abnormal return that's a
    known deterministic function of divergence, so the expected quintile
    pattern is known exactly.
    """
    divergence = np.linspace(-1, 1, n)
    if relationship == "decreasing":
        abnormal_return = -0.02 * divergence
    elif relationship == "u_shaped":
        abnormal_return = 0.02 * divergence**2
    else:
        raise ValueError(relationship)

    return pd.DataFrame(
        {
            "ticker": [f"T{i % 5}" for i in range(n)],
            "year": [2020 + i % 3 for i in range(n)],
            "management_qa_divergence": divergence,
            "abnormal_return_5d": abnormal_return,
        }
    )


# --- assign_divergence_quintiles ------------------------------------------


def test_assign_divergence_quintiles_raises_when_too_few_calls():
    df = _synthetic_calls(n=4)
    with pytest.raises(ValueError):
        assign_divergence_quintiles(df)


def test_assign_divergence_quintiles_forms_five_balanced_groups():
    df = _synthetic_calls(n=30)
    out = assign_divergence_quintiles(df)

    assert set(out["divergence_quintile"].unique()) == {1, 2, 3, 4, 5}
    counts = out["divergence_quintile"].value_counts()
    assert (counts == 6).all()  # 30 calls / 5 groups, no ties


def test_assign_divergence_quintiles_lowest_group_has_lowest_divergence():
    df = _synthetic_calls(n=30)
    out = assign_divergence_quintiles(df)

    q1_max = out.loc[out["divergence_quintile"] == 1, "management_qa_divergence"].max()
    q5_min = out.loc[out["divergence_quintile"] == 5, "management_qa_divergence"].min()
    assert q1_max < q5_min


def test_assign_divergence_quintiles_warns_on_heavy_ties():
    # 10 rows but only 2 distinct divergence values -> can't form 5 groups
    df = pd.DataFrame(
        {
            "management_qa_divergence": [0.0] * 5 + [1.0] * 5,
            "abnormal_return_5d": np.linspace(-0.1, 0.1, 10),
        }
    )
    with pytest.warns(UserWarning, match="too many tied"):
        out = assign_divergence_quintiles(df)
    assert out["divergence_quintile"].nunique() < 5


# --- quintile_return_summary / is_monotonic / long_short_spread ----------


def test_quintile_return_summary_matches_manual_groupby():
    df = _synthetic_calls(n=30)
    quintiled = assign_divergence_quintiles(df)

    summary = quintile_return_summary(quintiled, ["abnormal_return_5d"])

    for q in range(1, 6):
        expected_mean = quintiled.loc[quintiled["divergence_quintile"] == q, "abnormal_return_5d"].mean()
        actual_mean = summary.loc[summary["divergence_quintile"] == q, "abnormal_return_5d_mean"].iloc[0]
        assert actual_mean == pytest.approx(expected_mean)
        assert summary.loc[summary["divergence_quintile"] == q, "abnormal_return_5d_count"].iloc[0] == 6


def test_is_monotonic_detects_decreasing_relationship():
    df = _synthetic_calls(n=30, relationship="decreasing")
    quintiled = assign_divergence_quintiles(df)
    summary = quintile_return_summary(quintiled, ["abnormal_return_5d"])

    assert is_monotonic(summary, "abnormal_return_5d_mean", direction="decreasing")
    assert not is_monotonic(summary, "abnormal_return_5d_mean", direction="increasing")


def test_is_monotonic_detects_non_monotonic_u_shape():
    df = _synthetic_calls(n=30, relationship="u_shaped")
    quintiled = assign_divergence_quintiles(df)
    summary = quintile_return_summary(quintiled, ["abnormal_return_5d"])

    assert not is_monotonic(summary, "abnormal_return_5d_mean", direction="decreasing")
    assert not is_monotonic(summary, "abnormal_return_5d_mean", direction="increasing")


def test_long_short_spread_matches_hypothesis_direction():
    df = _synthetic_calls(n=30, relationship="decreasing")
    quintiled = assign_divergence_quintiles(df)
    summary = quintile_return_summary(quintiled, ["abnormal_return_5d"])

    spread = long_short_spread(summary, "abnormal_return_5d_mean")

    assert spread["low_quintile"] == 1
    assert spread["high_quintile"] == 5
    # low-divergence group has the highest (least negative) mean return here,
    # so the long-low/short-high spread should be positive
    assert spread["long_short_spread"] > 0
    assert spread["long_short_spread"] == pytest.approx(spread["low_mean_return"] - spread["high_mean_return"])


# --- winsorize_returns -----------------------------------------------------


def test_winsorize_returns_clips_extreme_value():
    df = pd.DataFrame({"abnormal_return_5d": [1.0, 2.0, 3.0, 4.0, 100.0]})
    out = winsorize_returns(df, "abnormal_return_5d", limits=(0.0, 0.2))
    assert out["abnormal_return_5d"].max() == pytest.approx(4.0)
    assert list(out["abnormal_return_5d"][:4]) == [1.0, 2.0, 3.0, 4.0]


def test_winsorize_returns_preserves_nan():
    df = pd.DataFrame({"abnormal_return_5d": [1.0, 2.0, float("nan"), 4.0, 100.0]})
    out = winsorize_returns(df, "abnormal_return_5d", limits=(0.0, 0.2))
    assert out["abnormal_return_5d"].isna().sum() == 1


# --- quintile_summary_by_group / conditional_long_short_spread ------------


def test_quintile_summary_by_group_breaks_out_by_year():
    df = _synthetic_calls(n=30)
    quintiled = assign_divergence_quintiles(df)

    by_year = quintile_summary_by_group(quintiled, "year", "abnormal_return_5d")

    assert set(by_year["year"].unique()) == set(quintiled["year"].unique())
    assert "abnormal_return_5d_mean" in by_year.columns


def test_conditional_long_short_spread_splits_by_mask():
    df = _synthetic_calls(n=30, relationship="decreasing")
    mask = df["ticker"].isin(["T0", "T1", "T2"])

    results = conditional_long_short_spread(df, mask, "abnormal_return_5d")

    assert "condition_true" in results
    assert "condition_false" in results
    assert "long_short_spread" in results["condition_true"]


# --- plotting (smoke tests) ------------------------------------------------


def test_plot_functions_return_figures_without_saving():
    df = _synthetic_calls(n=30)
    quintiled = assign_divergence_quintiles(df)
    summary = quintile_return_summary(quintiled, ["abnormal_return_5d"])

    fig1 = plot_divergence_scatter(df, "management_qa_divergence", "abnormal_return_5d")
    fig2 = plot_divergence_distribution(df, "management_qa_divergence")
    fig3 = plot_quintile_mean_returns(summary, "abnormal_return_5d_mean")

    for fig in (fig1, fig2, fig3):
        assert fig is not None
        plt.close(fig)
