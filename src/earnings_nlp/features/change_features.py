"""Prepared-to-Q&A change features, including the signature
`management_qa_divergence` feature (Phase 9).

Operates on the per-call sentiment table produced by
`earnings_nlp.features.finbert_features.aggregate_call_sentiment`.
"""

from __future__ import annotations

import re

import pandas as pd

_QUARTER_PATTERN = re.compile(r"(\d{4})Q([1-4])")


def _quarter_sort_key(quarter: str) -> tuple[int, int]:
    match = _QUARTER_PATTERN.fullmatch(quarter)
    if not match:
        raise ValueError(f"Unrecognized quarter format: {quarter!r}")
    year, q = match.groups()
    return int(year), int(q)


def add_divergence_features(call_df: pd.DataFrame) -> pd.DataFrame:
    """Add the same-call divergence/gap features (no cross-quarter context
    needed): management_qa_divergence, analyst_management_gap, ceo_cfo_gap.

    A large positive management_qa_divergence means management sounded
    more positive in its prepared script than during Q&A; near zero means
    tone was consistent; negative means management became more positive
    under questioning.
    """
    out = call_df.copy()
    out["management_qa_divergence"] = (
        out["prepared_management_sentiment"] - out["qa_management_sentiment"]
    )
    out["analyst_management_gap"] = (
        out["qa_management_sentiment"] - out["analyst_question_sentiment"]
    )
    out["ceo_cfo_gap"] = out["ceo_sentiment"] - out["cfo_sentiment"]
    return out


def add_quarterly_change_features(call_df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-quarter change features that require each ticker's calls
    to be ordered chronologically: quarterly_sentiment_change (change in
    overall_sentiment vs. that ticker's previous call) and
    qa_negativity_change (change in qa_negative_probability vs. that
    ticker's previous call). The first available quarter for a ticker has
    no prior call, so these are NaN there.
    """
    out = call_df.copy()
    out["_sort_key"] = out["quarter"].map(_quarter_sort_key)
    out = out.sort_values(["ticker", "_sort_key"]).drop(columns="_sort_key").reset_index(drop=True)

    out["quarterly_sentiment_change"] = out.groupby("ticker")["overall_sentiment"].diff()
    out["qa_negativity_change"] = out.groupby("ticker")["qa_negative_probability"].diff()

    return out


def build_divergence_table(call_df: pd.DataFrame) -> pd.DataFrame:
    """Full Phase 9 feature set: same-call divergence/gap features plus
    cross-quarter change features, with `prepared_sentiment`/`qa_sentiment`
    aliases matching the final research table's naming (Phase 9 spec).
    """
    out = add_divergence_features(call_df)
    out = add_quarterly_change_features(out)
    out["prepared_sentiment"] = out["prepared_management_sentiment"]
    out["qa_sentiment"] = out["qa_management_sentiment"]
    return out
