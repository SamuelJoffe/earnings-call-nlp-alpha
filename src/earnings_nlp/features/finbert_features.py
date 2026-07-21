"""FinBERT-based sentiment features for earnings call transcripts (Phase 8).

Each turn's text is split into <=510-token chunks (see
`earnings_nlp.processing.chunk_text`), each chunk is scored by FinBERT for
positive/negative/neutral probability, and a continuous
`sentiment_score = positive_probability - negative_probability` is derived
per chunk. Chunk-level scores are then aggregated per call into the same
prepared/qa_management/analyst/ceo/cfo groups used for the Phase 7
linguistic features, plus two whole-call features: sentiment_dispersion
and negative_chunk_percentage.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from earnings_nlp.features.groups import GROUP_MASKS
from earnings_nlp.processing.chunk_text import chunk_text, get_classifier

LABELS = ("positive", "negative", "neutral")

# Phase 9's divergence formulas reference these exact feature names
# (prepared_management_sentiment, qa_management_sentiment,
# analyst_question_sentiment, ceo_sentiment, cfo_sentiment).
_FEATURE_NAME = {
    "prepared": "prepared_management",
    "qa_management": "qa_management",
    "analyst": "analyst_question",
    "ceo": "ceo",
    "cfo": "cfo",
}


def score_chunks(chunks: list[str], classifier: Callable | None = None) -> list[dict]:
    """Run FinBERT on a batch of chunks, returning one probability dict per
    chunk. `classifier` can be injected (e.g. in tests) to avoid loading the
    real model; it must behave like a `transformers` `top_k=None`
    text-classification pipeline.
    """
    if not chunks:
        return []
    classifier = classifier or get_classifier()
    raw_results = classifier(chunks)

    scored = []
    for result in raw_results:
        probs = {r["label"].lower(): r["score"] for r in result}
        scored.append({f"{label}_probability": probs.get(label, 0.0) for label in LABELS})
    return scored


def add_chunk_sentiment(df: pd.DataFrame, classifier: Callable | None = None) -> pd.DataFrame:
    """Explode a parsed transcript dataframe into one row per <=510-token
    chunk, with FinBERT sentiment probabilities, sentiment_score, and the
    chunk's predicted label (argmax of the three probabilities).
    """
    records = []
    for _, row in df.iterrows():
        chunks = chunk_text(row["text"])
        if not chunks:
            continue
        for i, (chunk, scores) in enumerate(zip(chunks, score_chunks(chunks, classifier))):
            record = row.to_dict()
            record["chunk_index"] = i
            record["chunk_text"] = chunk
            record.update(scores)
            record["sentiment_score"] = (
                scores["positive_probability"] - scores["negative_probability"]
            )
            record["predicted_label"] = max(LABELS, key=lambda label: scores[f"{label}_probability"])
            records.append(record)

    return pd.DataFrame(records)


def aggregate_call_sentiment(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chunk-level FinBERT sentiment into one row per
    (ticker, quarter) call.
    """
    rows = []
    for (ticker, quarter), call_df in chunk_df.groupby(["ticker", "quarter"], sort=False):
        row = {"ticker": ticker, "quarter": quarter}

        for prefix, mask_fn in GROUP_MASKS.items():
            subset = call_df[mask_fn(call_df)]
            feature_name = f"{_FEATURE_NAME[prefix]}_sentiment"
            row[feature_name] = subset["sentiment_score"].mean() if len(subset) else float("nan")

        if len(call_df):
            row["sentiment_dispersion"] = call_df["sentiment_score"].std(ddof=0)
            row["negative_chunk_percentage"] = 100 * (call_df["predicted_label"] == "negative").mean()
        else:
            row["sentiment_dispersion"] = float("nan")
            row["negative_chunk_percentage"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)
