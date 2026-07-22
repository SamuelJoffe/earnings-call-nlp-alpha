import math

import numpy as np
import pandas as pd
import pytest

from earnings_nlp.backtest.event_returns import (
    build_event_return_table,
    compute_event_return,
    price_series_returns,
    signal_entry_date,
)


def _trading_days(start="2024-01-01", periods=40):
    return pd.bdate_range(start=start, periods=periods)


# --- signal_entry_date ---------------------------------------------------


def test_signal_entry_date_after_close_uses_next_trading_day():
    days = _trading_days()
    entry = signal_entry_date("2024-01-03", "after_close", days)  # a Wednesday
    assert entry == pd.Timestamp("2024-01-04")


def test_signal_entry_date_before_open_uses_same_day():
    days = _trading_days()
    entry = signal_entry_date("2024-01-03", "before_open", days)
    assert entry == pd.Timestamp("2024-01-03")


def test_signal_entry_date_during_hours_uses_same_day():
    days = _trading_days()
    entry = signal_entry_date("2024-01-03", "during_hours", days)
    assert entry == pd.Timestamp("2024-01-03")


def test_signal_entry_date_after_close_skips_weekend():
    days = _trading_days()
    entry = signal_entry_date("2024-01-05", "after_close", days)  # a Friday
    assert entry == pd.Timestamp("2024-01-08")  # following Monday, not Sat/Sun


def test_signal_entry_date_invalid_timing_raises():
    with pytest.raises(ValueError):
        signal_entry_date("2024-01-03", "at_lunch", _trading_days())


def test_signal_entry_date_raises_when_no_trading_day_after():
    days = pd.bdate_range("2024-01-01", periods=3)  # ends 2024-01-03
    with pytest.raises(ValueError):
        signal_entry_date("2024-01-03", "after_close", days)


# --- price_series_returns -------------------------------------------------


def test_price_series_returns_known_values_and_nan_past_end():
    days = _trading_days(periods=10)
    prices = pd.Series(100 + np.arange(10), index=days)  # 100..109
    entry_date = days[3]  # price 103

    out = price_series_returns(prices, entry_date, horizons=(1, 5, 20))

    assert out["forward_return_1d"] == pytest.approx(104 / 103 - 1)
    assert out["forward_return_5d"] == pytest.approx(108 / 103 - 1)
    assert math.isnan(out["forward_return_20d"])  # index 23 is past the end


def test_price_series_returns_raises_if_entry_date_missing():
    days = _trading_days(periods=10)
    prices = pd.Series(100 + np.arange(10), index=days)
    with pytest.raises(ValueError):
        price_series_returns(prices, pd.Timestamp("2099-01-01"))


# --- compute_event_return -------------------------------------------------


@pytest.fixture
def ticker_and_benchmark():
    days = _trading_days(periods=10)
    ticker_prices = pd.Series([100, 102, 101, 105, 110, 108, 115, 120, 118, 125], index=days)
    benchmark_prices = pd.Series([200, 201, 202, 203, 204, 205, 206, 207, 208, 209], index=days)
    return days, ticker_prices, benchmark_prices


def test_compute_event_return_abnormal_return(ticker_and_benchmark):
    days, ticker_prices, benchmark_prices = ticker_and_benchmark

    result = compute_event_return(
        ticker_prices, benchmark_prices, earnings_date=days[2], timing="after_close"
    )

    assert result["entry_date"] == days[3]
    expected_ticker_1d = ticker_prices.iloc[4] / ticker_prices.iloc[3] - 1
    expected_benchmark_1d = benchmark_prices.iloc[4] / benchmark_prices.iloc[3] - 1
    assert result["forward_return_1d"] == pytest.approx(expected_ticker_1d)
    assert result["abnormal_return_1d"] == pytest.approx(expected_ticker_1d - expected_benchmark_1d)


def test_compute_event_return_before_open_uses_same_day_entry(ticker_and_benchmark):
    days, ticker_prices, benchmark_prices = ticker_and_benchmark

    result = compute_event_return(
        ticker_prices, benchmark_prices, earnings_date=days[3], timing="before_open"
    )
    assert result["entry_date"] == days[3]


def test_compute_event_return_raises_if_entry_missing_from_benchmark(ticker_and_benchmark):
    days, ticker_prices, benchmark_prices = ticker_and_benchmark
    benchmark_missing_entry = benchmark_prices.drop(days[3])

    with pytest.raises(ValueError):
        compute_event_return(
            ticker_prices, benchmark_missing_entry, earnings_date=days[2], timing="after_close"
        )


def test_compute_event_return_nan_abnormal_when_forward_return_is_nan(ticker_and_benchmark):
    days, ticker_prices, benchmark_prices = ticker_and_benchmark
    # near the end of the series, the 20-day forward return doesn't exist
    result = compute_event_return(
        ticker_prices, benchmark_prices, earnings_date=days[6], timing="before_open"
    )
    assert math.isnan(result["forward_return_20d"])
    assert math.isnan(result["abnormal_return_20d"])


# --- build_event_return_table --------------------------------------------


def test_build_event_return_table_one_row_per_call(ticker_and_benchmark):
    days, ticker_prices, benchmark_prices = ticker_and_benchmark
    other_prices = ticker_prices * 2  # a distinct second "ticker"

    calls = [
        dict(ticker="AAPL", quarter="2024Q1", earnings_date=days[2], timing="after_close"),
        dict(ticker="MSFT", quarter="2024Q1", earnings_date=days[1], timing="before_open"),
    ]
    prices_by_ticker = {"AAPL": ticker_prices, "MSFT": other_prices}

    out = build_event_return_table(calls, prices_by_ticker, benchmark_prices)

    assert len(out) == 2
    assert set(out["ticker"]) == {"AAPL", "MSFT"}
    assert "forward_return_1d" in out.columns
    assert "abnormal_return_5d" in out.columns
