"""Simple interpretable linguistic features, computed directly on turn text
before any FinBERT/LLM scoring is introduced (Phase 7 baseline).

The word lists below are a small, hand-curated baseline lexicon inspired by
the Loughran-McDonald financial-sentiment categories (positive/negative/
uncertainty) — not the full LM dictionary. They exist to prove the pipeline
works end to end; swap in the full LM dictionary later if this baseline
looks useful enough to refine.
"""

from __future__ import annotations

import re

import pandas as pd

_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
_WORD_PATTERN = re.compile(r"[A-Za-z']+")
_NUMERIC_TOKEN_PATTERN = re.compile(r"\b\S*\d\S*\b")

FIRST_PERSON_PLURAL_WORDS = {"we", "us", "our", "ours", "ourselves"}

POSITIVE_WORDS = {
    "growth", "grew", "strong", "strength", "strengthen", "record",
    "increase", "increased", "improve", "improved", "improvement", "exceed",
    "exceeded", "beat", "robust", "solid", "favorable", "opportunity",
    "opportunities", "success", "successful", "accelerate", "accelerating",
    "momentum", "confident", "confidence", "gain", "gains", "outperform",
    "upside", "progress", "positive", "expand", "expanding", "expansion",
    "win", "wins", "winning", "healthy", "optimistic", "pleased",
    "delighted", "great", "excellent", "outstanding", "leadership",
}

NEGATIVE_WORDS = {
    "decline", "declined", "declining", "decrease", "decreased", "weak",
    "weakness", "headwind", "headwinds", "challenge", "challenges",
    "challenging", "difficult", "pressure", "pressures", "loss", "losses",
    "miss", "missed", "below", "slowdown", "slowing", "softness", "soft",
    "negative", "concern", "concerns", "concerned", "disappointing",
    "disappointed", "impairment", "litigation", "volatile", "adverse",
    "warn", "warning", "shortfall", "layoffs", "restructuring",
}

UNCERTAINTY_WORDS = {
    "may", "might", "could", "uncertain", "uncertainty", "uncertainties",
    "approximately", "possibly", "perhaps", "likely", "unlikely",
    "estimate", "estimates", "estimated", "assume", "assumes", "assuming",
    "assumption", "assumptions", "believe", "believes", "depend", "depends",
    "depending", "risk", "risks", "risky", "volatility", "unpredictable",
    "roughly", "around", "potential", "potentially", "expect", "expects",
    "expected", "anticipate", "anticipated",
}

FEATURE_COLUMNS = [
    "word_count",
    "average_sentence_length",
    "question_count",
    "positive_word_count",
    "negative_word_count",
    "uncertainty_word_count",
    "first_person_plural_count",
    "numeric_token_count",
]

MANAGEMENT_ROLES = {"CEO", "CFO", "Other Management"}

# Category name -> boolean mask over a turn-level dataframe.
_GROUPS = {
    "prepared": lambda df: (df["section"] == "prepared") & df["role"].isin(MANAGEMENT_ROLES),
    "qa_management": lambda df: (df["section"] == "qa") & df["role"].isin(MANAGEMENT_ROLES),
    "analyst": lambda df: (df["section"] == "qa") & (df["role"] == "Analyst"),
    "ceo": lambda df: df["role"] == "CEO",
    "cfo": lambda df: df["role"] == "CFO",
}


def _words(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


def _sentence_count(text: str) -> int:
    sentences = [s for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
    return max(len(sentences), 1)


def compute_turn_features(text: str) -> dict:
    """Compute the eight baseline linguistic features for one turn of text."""
    text = text or ""
    words = _words(text)
    word_count = len(words)
    sentence_count = _sentence_count(text)

    return {
        "word_count": word_count,
        "average_sentence_length": word_count / sentence_count,
        "question_count": text.count("?"),
        "positive_word_count": sum(1 for w in words if w in POSITIVE_WORDS),
        "negative_word_count": sum(1 for w in words if w in NEGATIVE_WORDS),
        "uncertainty_word_count": sum(1 for w in words if w in UNCERTAINTY_WORDS),
        "first_person_plural_count": sum(1 for w in words if w in FIRST_PERSON_PLURAL_WORDS),
        "numeric_token_count": len(_NUMERIC_TOKEN_PATTERN.findall(text)),
    }


def add_turn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a parsed transcript dataframe with one linguistic
    feature column added per turn.
    """
    features = df["text"].apply(compute_turn_features).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), features[FEATURE_COLUMNS]], axis=1)


def _aggregate_group(call_df: pd.DataFrame, mask: pd.Series, prefix: str) -> dict:
    subset = call_df[mask]
    word_count = int(subset["word_count"].sum())

    out = {
        f"{prefix}_turn_count": len(subset),
        f"{prefix}_word_count": word_count,
        f"{prefix}_question_count": int(subset["question_count"].sum()),
    }

    if len(subset) == 0:
        out[f"{prefix}_avg_sentence_length"] = float("nan")
    else:
        out[f"{prefix}_avg_sentence_length"] = subset["average_sentence_length"].mean()

    rate_source = {
        "positive_rate": "positive_word_count",
        "negative_rate": "negative_word_count",
        "uncertainty_rate": "uncertainty_word_count",
        "first_person_plural_rate": "first_person_plural_count",
        "numeric_rate": "numeric_token_count",
    }
    for rate_name, count_col in rate_source.items():
        if word_count == 0:
            out[f"{prefix}_{rate_name}"] = float("nan")
        else:
            out[f"{prefix}_{rate_name}"] = 100 * subset[count_col].sum() / word_count

    return out


def aggregate_call_features(turn_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate turn-level linguistic features into one row per
    (ticker, quarter) call, broken out by prepared remarks, management Q&A
    answers, analyst questions, CEO speech, and CFO speech (all rates are
    per 100 words so they're comparable across sections of different
    length). Also adds the prepared-to-Q&A change features.
    """
    if "word_count" not in turn_df.columns:
        turn_df = add_turn_features(turn_df)

    rows = []
    for (ticker, quarter), call_df in turn_df.groupby(["ticker", "quarter"], sort=False):
        row = {"ticker": ticker, "quarter": quarter}
        for prefix, mask_fn in _GROUPS.items():
            row.update(_aggregate_group(call_df, mask_fn(call_df), prefix))

        row["qa_length_change"] = (
            row["qa_management_avg_sentence_length"] - row["prepared_avg_sentence_length"]
        )
        row["uncertainty_change"] = (
            row["qa_management_uncertainty_rate"] - row["prepared_uncertainty_rate"]
        )
        rows.append(row)

    return pd.DataFrame(rows)
