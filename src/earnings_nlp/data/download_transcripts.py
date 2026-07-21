"""Download earnings call transcripts from Alpha Vantage's
EARNINGS_CALL_TRANSCRIPT endpoint and cache them locally as JSON.

Usage:
    python -m earnings_nlp.data.download_transcripts
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv
import os

from earnings_nlp.utils.paths import CONFIG_DIR, DOWNLOAD_LOG, ROOT, TRANSCRIPTS_RAW

load_dotenv(ROOT / ".env")

BASE_URL = "https://www.alphavantage.co/query"
FUNCTION = "EARNINGS_CALL_TRANSCRIPT"


def load_config() -> dict:
    with open(CONFIG_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def transcript_path(ticker: str, quarter: str) -> Path:
    return TRANSCRIPTS_RAW / f"{ticker}_{quarter}.json"


def log_download(ticker: str, quarter: str, status: str, detail: str = "") -> None:
    DOWNLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not DOWNLOAD_LOG.exists()
    with open(DOWNLOAD_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "ticker", "quarter", "status", "detail"])
        writer.writerow([dt.datetime.now().isoformat(timespec="seconds"), ticker, quarter, status, detail])


def fetch_transcript(ticker: str, quarter: str, api_key: str) -> dict:
    params = {
        "function": FUNCTION,
        "symbol": ticker,
        "quarter": quarter,
        "apikey": api_key,
    }
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def is_error_response(payload: dict) -> str | None:
    """Return an error message if the payload looks like an API error, else None."""
    for key in ("Error Message", "Note", "Information"):
        if key in payload:
            return payload[key]
    if not payload.get("transcript"):
        return "response contained no 'transcript' field"
    return None


def download_transcripts(calls: list[dict], sleep_seconds: float = 15.0) -> None:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ALPHA_VANTAGE_API_KEY not set. Add it to a local .env file "
            "(see .env.example)."
        )

    TRANSCRIPTS_RAW.mkdir(parents=True, exist_ok=True)

    for i, call in enumerate(calls):
        ticker, quarter = call["ticker"], call["quarter"]
        out_path = transcript_path(ticker, quarter)

        if out_path.exists():
            print(f"[skip] {ticker} {quarter} already downloaded")
            log_download(ticker, quarter, "skipped", "already downloaded")
            continue

        print(f"[fetch] {ticker} {quarter}")
        try:
            payload = fetch_transcript(ticker, quarter, api_key)
        except requests.RequestException as exc:
            print(f"[error] {ticker} {quarter}: {exc}")
            log_download(ticker, quarter, "error", str(exc))
            continue

        error = is_error_response(payload)
        if error:
            print(f"[error] {ticker} {quarter}: {error}")
            log_download(ticker, quarter, "error", error)
            continue

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[saved] {out_path}")
        log_download(ticker, quarter, "ok")

        if i < len(calls) - 1:
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    config = load_config()
    download_transcripts(
        config["milestone_calls"],
        sleep_seconds=config["data_source"]["request_sleep_seconds"],
    )
