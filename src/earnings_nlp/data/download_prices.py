"""Download daily adjusted close prices for the milestone calls' tickers
plus the market benchmark (SPY), used to compute event returns (Phase 10).

Usage:
    python -m earnings_nlp.data.download_prices
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from earnings_nlp.data.download_transcripts import load_config
from earnings_nlp.utils.paths import PRICES_RAW


def price_path(ticker: str) -> Path:
    return PRICES_RAW / f"{ticker}.csv"


def download_ticker_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download one ticker's daily adjusted close prices between `start`
    and `end` (YYYY-MM-DD) and cache to data/raw/prices/TICKER.csv.
    """
    history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if history.empty:
        raise RuntimeError(f"no price data returned for {ticker} ({start} to {end})")

    df = history[["Close"]].rename(columns={"Close": "adj_close"})
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df = df.reset_index()
    df["ticker"] = ticker

    PRICES_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(price_path(ticker), index=False)
    return df


def download_prices(tickers: list[str], earnings_dates: list[str], benchmark_ticker: str) -> None:
    """Download prices for each ticker plus the benchmark, spanning a
    window wide enough to cover 20 trading days after the latest call and
    a buffer before the earliest call.
    """
    parsed_dates = pd.to_datetime(earnings_dates)
    start = (parsed_dates.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (parsed_dates.max() + pd.Timedelta(days=45)).strftime("%Y-%m-%d")

    all_tickers = sorted(set(tickers) | {benchmark_ticker})
    for ticker in all_tickers:
        print(f"[fetch] {ticker} ({start} to {end})")
        df = download_ticker_prices(ticker, start, end)
        print(f"[saved] {price_path(ticker)} ({len(df)} rows)")


if __name__ == "__main__":
    config = load_config()
    calls = config["milestone_calls"]
    download_prices(
        tickers=[c["ticker"] for c in calls],
        earnings_dates=[c["earnings_date"] for c in calls],
        benchmark_ticker=config["price_source"]["benchmark_ticker"],
    )
