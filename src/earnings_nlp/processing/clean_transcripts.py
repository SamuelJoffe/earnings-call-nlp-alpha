"""Turn a raw Alpha Vantage transcript JSON payload into a clean,
one-row-per-turn dataframe with speaker-role and section labels.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from earnings_nlp.processing.classify_speakers import classify_role, label_sections

MIN_WORDS = 4
_PAGE_ARTIFACT_PATTERN = re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def load_raw_transcript(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _normalize_text(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text or "").strip()


def _is_substantive(role: str, text: str) -> bool:
    """Drop operator instructions, empty turns, and very short filler."""
    if role == "Operator":
        return False
    if not text:
        return False
    if _PAGE_ARTIFACT_PATTERN.match(text):
        return False
    word_count = len(text.split())
    if word_count < MIN_WORDS and "?" not in text:
        return False
    return True


def parse_transcript(ticker: str, quarter: str, raw_json: dict) -> pd.DataFrame:
    """Build the clean per-turn dataframe for a single earnings call.

    Columns: ticker, quarter, speaker, title, section, role, text
    """
    turns = raw_json.get("transcript", [])
    sections = label_sections(turns)

    rows = []
    seen_text: set[str] = set()

    for turn, section in zip(turns, sections):
        speaker = turn.get("speaker", "") or ""
        title = turn.get("title", "") or ""
        text = _normalize_text(turn.get("content", ""))
        role = classify_role(speaker, title)

        if not _is_substantive(role, text):
            continue

        dedup_key = text.lower()
        if dedup_key in seen_text:
            continue
        seen_text.add(dedup_key)

        rows.append(
            {
                "ticker": ticker,
                "quarter": quarter,
                "speaker": speaker,
                "title": title,
                "role": role,
                "section": section,
                "text": text,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["ticker", "quarter", "speaker", "title", "role", "section", "text"],
    )


def parse_transcript_file(path: str | Path) -> pd.DataFrame:
    """Parse a transcript JSON file, inferring ticker/quarter from its name
    (expects the `download_transcripts.py` naming convention TICKER_QUARTER.json).
    """
    path = Path(path)
    ticker, quarter = path.stem.split("_", 1)
    raw_json = load_raw_transcript(path)
    return parse_transcript(ticker, quarter, raw_json)


def parse_transcript_files(paths: list[str | Path]) -> pd.DataFrame:
    """Parse multiple transcript files into a single combined dataframe."""
    frames = [parse_transcript_file(p) for p in paths]
    if not frames:
        return pd.DataFrame(
            columns=["ticker", "quarter", "speaker", "title", "role", "section", "text"]
        )
    return pd.concat(frames, ignore_index=True)
