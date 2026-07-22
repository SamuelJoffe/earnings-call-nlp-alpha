"""Compute forward and market-adjusted event returns per earnings call
(Phase 10).

Critical timing rule: a call reported after that day's market close
cannot use that day's close as though the transcript were already public
-- the first close that could possibly reflect the call is the *next*
trading day's close. A call reported before the open, or during trading
hours, uses that same day's close as the signal-entry point, since the
regular session that day already had the chance to react to it.

    R_{i,t->t+k}  = P_{i,t+k} / P_{i,t} - 1
    AR_{i,t->t+k} = R_{i,t->t+k} - R_{m,t->t+k}

evaluated for k = 1, 5, and 20 trading days.
"""

from __future__ import annotations

import pandas as pd

from earnings_nlp.utils.paths import PRICES_RAW

HORIZONS = (1, 5, 20)
VALID_TIMINGS = ("before_open", "during_hours", "after_close")


def load_price_series(ticker: str, prices_dir=PRICES_RAW) -> pd.Series:
    """Load a cached data/raw/prices/TICKER.csv into a date-indexed,
    ascending adj_close Series.
    """
    df = pd.read_csv(prices_dir / f"{ticker}.csv", parse_dates=["date"])
    return df.set_index("date")["adj_close"].sort_index()


def signal_entry_date(earnings_date, timing: str, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the first trading-day date whose close could reflect the
    call's information.
    """
    if timing not in VALID_TIMINGS:
        raise ValueError(f"unknown timing: {timing!r}, expected one of {VALID_TIMINGS}")

    earnings_date = pd.Timestamp(earnings_date)

    if timing == "after_close":
        candidates = trading_days[trading_days > earnings_date]
        if candidates.empty:
            raise ValueError(f"no trading day after {earnings_date.date()}")
    else:  # before_open or during_hours: same day's close already reacted
        candidates = trading_days[trading_days >= earnings_date]
        if candidates.empty:
            raise ValueError(f"no trading day on/after {earnings_date.date()}")

    return candidates[0]


def price_series_returns(prices: pd.Series, entry_date: pd.Timestamp, horizons=HORIZONS) -> dict:
    """Forward returns R_{t->t+k} for each k in `horizons`, keyed
    `forward_return_{k}d`. NaN when there isn't enough subsequent data.
    """
    if entry_date not in prices.index:
        raise ValueError(f"{entry_date.date()} is not in the price series")

    entry_idx = prices.index.get_loc(entry_date)
    entry_price = prices.iloc[entry_idx]

    out = {}
    for k in horizons:
        target_idx = entry_idx + k
        if target_idx >= len(prices):
            out[f"forward_return_{k}d"] = float("nan")
        else:
            out[f"forward_return_{k}d"] = prices.iloc[target_idx] / entry_price - 1
    return out


def compute_event_return(
    ticker_prices: pd.Series,
    benchmark_prices: pd.Series,
    earnings_date,
    timing: str,
    horizons=HORIZONS,
) -> dict:
    """Signal-entry date, forward returns, and market-adjusted (abnormal)
    returns for one earnings call.
    """
    entry_date = signal_entry_date(earnings_date, timing, ticker_prices.index)
    if entry_date not in benchmark_prices.index:
        raise ValueError(f"entry date {entry_date.date()} missing from benchmark price series")

    ticker_returns = price_series_returns(ticker_prices, entry_date, horizons)
    benchmark_returns = price_series_returns(benchmark_prices, entry_date, horizons)

    result = {"entry_date": entry_date}
    for k in horizons:
        col = f"forward_return_{k}d"
        result[col] = ticker_returns[col]
        both_present = pd.notna(ticker_returns[col]) and pd.notna(benchmark_returns[col])
        result[f"abnormal_return_{k}d"] = (
            ticker_returns[col] - benchmark_returns[col] if both_present else float("nan")
        )
    return result


def build_event_return_table(
    calls: list[dict],
    prices_by_ticker: dict[str, pd.Series],
    benchmark_prices: pd.Series,
    horizons=HORIZONS,
) -> pd.DataFrame:
    """One row per call: ticker, quarter, earnings_date, timing,
    signal-entry date, and forward/abnormal returns at each horizon.
    """
    rows = []
    for call in calls:
        ticker = call["ticker"]
        row = {
            "ticker": ticker,
            "quarter": call["quarter"],
            "earnings_date": call["earnings_date"],
            "timing": call["timing"],
        }
        row.update(
            compute_event_return(
                prices_by_ticker[ticker],
                benchmark_prices,
                call["earnings_date"],
                call["timing"],
                horizons,
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)
