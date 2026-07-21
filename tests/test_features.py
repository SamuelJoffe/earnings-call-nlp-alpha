import math

import pandas as pd
import pytest

from earnings_nlp.features.finbert_features import (
    add_chunk_sentiment,
    aggregate_call_sentiment,
    score_chunks,
)
from earnings_nlp.features.linguistic_features import (
    add_turn_features,
    aggregate_call_features,
    compute_turn_features,
)
from earnings_nlp.processing.chunk_text import chunk_text, get_tokenizer


def test_compute_turn_features_known_counts():
    text = (
        "We believe our strong growth will continue. "
        "However, there could be some risk to margins in Q3? "
        "Revenue grew 15% to $2.5 billion."
    )
    features = compute_turn_features(text)

    assert features["word_count"] == 21
    assert features["question_count"] == 1
    assert features["positive_word_count"] == 3  # strong, growth, grew
    assert features["negative_word_count"] == 0
    assert features["uncertainty_word_count"] == 3  # believe, could, risk
    assert features["first_person_plural_count"] == 2  # we, our
    assert features["numeric_token_count"] == 3  # Q3, 15%, $2.5


def test_compute_turn_features_empty_text():
    features = compute_turn_features("")
    assert features["word_count"] == 0
    assert features["average_sentence_length"] == 0
    assert features["question_count"] == 0


def test_average_sentence_length():
    text = "One two three. Four five."
    features = compute_turn_features(text)
    assert features["word_count"] == 5
    assert features["average_sentence_length"] == pytest.approx(2.5)


def test_add_turn_features_preserves_row_count():
    df = pd.DataFrame(
        {
            "ticker": ["TEST", "TEST"],
            "quarter": ["2024Q1", "2024Q1"],
            "speaker": ["A", "B"],
            "title": ["CEO", "Analyst"],
            "role": ["CEO", "Analyst"],
            "section": ["prepared", "qa"],
            "text": ["We are confident in strong growth.", "Is there any risk to guidance?"],
        }
    )
    out = add_turn_features(df)
    assert len(out) == len(df)
    assert "word_count" in out.columns
    assert out.loc[1, "question_count"] == 1


@pytest.fixture
def call_df():
    rows = [
        # AAPL 2024Q1: prepared CEO remark, prepared CFO remark,
        # analyst question, CFO qa answer
        dict(ticker="AAPL", quarter="2024Q1", speaker="Tim", title="CEO", role="CEO",
             section="prepared", text="We delivered strong record growth and strong momentum this quarter."),
        dict(ticker="AAPL", quarter="2024Q1", speaker="Luca", title="CFO", role="CFO",
             section="prepared", text="Our margins improved and revenue increased year over year."),
        dict(ticker="AAPL", quarter="2024Q1", speaker="Analyst", title="Analyst", role="Analyst",
             section="qa", text="Can you comment on the sustainability of that margin expansion?"),
        dict(ticker="AAPL", quarter="2024Q1", speaker="Luca", title="CFO", role="CFO",
             section="qa", text="We believe the improvement could be uncertain given potential risk."),
    ]
    return pd.DataFrame(rows)


def test_aggregate_call_features_group_membership(call_df):
    out = aggregate_call_features(call_df)
    assert len(out) == 1
    row = out.iloc[0]

    assert row["prepared_turn_count"] == 2
    assert row["qa_management_turn_count"] == 1
    assert row["analyst_turn_count"] == 1
    assert row["ceo_turn_count"] == 1
    assert row["cfo_turn_count"] == 2


def test_aggregate_call_features_change_columns(call_df):
    out = aggregate_call_features(call_df)
    row = out.iloc[0]

    assert row["qa_length_change"] == pytest.approx(
        row["qa_management_avg_sentence_length"] - row["prepared_avg_sentence_length"]
    )
    assert row["uncertainty_change"] == pytest.approx(
        row["qa_management_uncertainty_rate"] - row["prepared_uncertainty_rate"]
    )
    # the qa CFO answer has more uncertainty words per 100 words than the
    # prepared remarks, so the change should be positive
    assert row["uncertainty_change"] > 0


def test_aggregate_call_features_empty_group_is_nan():
    # a call with no analyst turns at all
    rows = [
        dict(ticker="X", quarter="2024Q1", speaker="A", title="CEO", role="CEO",
             section="prepared", text="We are pleased with strong results."),
    ]
    out = aggregate_call_features(pd.DataFrame(rows))
    row = out.iloc[0]
    assert row["analyst_turn_count"] == 0
    assert math.isnan(row["analyst_positive_rate"])


# --- Phase 8: FinBERT sentiment ---------------------------------------


def test_chunk_text_splits_on_token_boundaries():
    tokenizer = get_tokenizer()
    long_text = "Revenue grew and margins improved. " * 300

    chunks = chunk_text(long_text, max_tokens=510)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert len(tokenizer.encode(chunk, add_special_tokens=False)) == 510
    assert len(tokenizer.encode(chunks[-1], add_special_tokens=False)) <= 510


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def _fake_classifier(texts):
    """Stand in for the real FinBERT pipeline: positive/negative/neutral
    scores are picked from a keyword in each input so tests are deterministic
    and don't need to load the actual model.
    """
    results = []
    for text in texts:
        lowered = text.lower()
        if "posneg_tie" in lowered:
            scores = {"positive": 0.5, "negative": 0.5, "neutral": 0.0}
        elif "negative" in lowered:
            scores = {"positive": 0.05, "negative": 0.9, "neutral": 0.05}
        elif "positive" in lowered:
            scores = {"positive": 0.9, "negative": 0.05, "neutral": 0.05}
        else:
            scores = {"positive": 0.1, "negative": 0.1, "neutral": 0.8}
        results.append([{"label": k, "score": v} for k, v in scores.items()])
    return results


def test_score_chunks_computes_expected_probabilities():
    scored = score_chunks(["this is a positive chunk", "this is a negative chunk"], _fake_classifier)

    assert scored[0]["positive_probability"] == pytest.approx(0.9)
    assert scored[0]["negative_probability"] == pytest.approx(0.05)
    assert scored[1]["negative_probability"] == pytest.approx(0.9)


def test_score_chunks_empty_list():
    assert score_chunks([], _fake_classifier) == []


def test_add_chunk_sentiment_one_row_per_chunk(monkeypatch):
    # force two chunks out of a single turn regardless of its actual length
    monkeypatch.setattr(
        "earnings_nlp.features.finbert_features.chunk_text",
        lambda text: ["positive chunk", "negative chunk"],
    )
    df = pd.DataFrame(
        [
            dict(ticker="AAPL", quarter="2024Q1", speaker="Tim", title="CEO", role="CEO",
                 section="prepared", text="irrelevant, chunking is mocked"),
        ]
    )

    out = add_chunk_sentiment(df, classifier=_fake_classifier)

    assert len(out) == 2
    assert list(out["chunk_index"]) == [0, 1]
    assert out.loc[0, "predicted_label"] == "positive"
    assert out.loc[1, "predicted_label"] == "negative"
    assert out.loc[0, "sentiment_score"] == pytest.approx(0.9 - 0.05)
    assert out.loc[1, "sentiment_score"] == pytest.approx(0.05 - 0.9)
    # original turn metadata carried onto every chunk row
    assert (out["role"] == "CEO").all()


@pytest.fixture
def chunk_df():
    rows = [
        dict(ticker="AAPL", quarter="2024Q1", role="CEO", section="prepared",
             sentiment_score=0.8, predicted_label="positive"),
        dict(ticker="AAPL", quarter="2024Q1", role="CFO", section="prepared",
             sentiment_score=0.6, predicted_label="positive"),
        dict(ticker="AAPL", quarter="2024Q1", role="Analyst", section="qa",
             sentiment_score=-0.2, predicted_label="negative"),
        dict(ticker="AAPL", quarter="2024Q1", role="CFO", section="qa",
             sentiment_score=-0.4, predicted_label="negative"),
        dict(ticker="AAPL", quarter="2024Q1", role="CFO", section="qa",
             sentiment_score=0.0, predicted_label="neutral"),
    ]
    return pd.DataFrame(rows)


def test_aggregate_call_sentiment_group_means(chunk_df):
    out = aggregate_call_sentiment(chunk_df)
    row = out.iloc[0]

    assert row["prepared_management_sentiment"] == pytest.approx((0.8 + 0.6) / 2)
    assert row["qa_management_sentiment"] == pytest.approx((-0.4 + 0.0) / 2)
    assert row["analyst_question_sentiment"] == pytest.approx(-0.2)
    assert row["ceo_sentiment"] == pytest.approx(0.8)
    assert row["cfo_sentiment"] == pytest.approx((0.6 - 0.4 + 0.0) / 3)


def test_aggregate_call_sentiment_dispersion_and_negative_percentage(chunk_df):
    out = aggregate_call_sentiment(chunk_df)
    row = out.iloc[0]

    expected_std = pd.Series([0.8, 0.6, -0.2, -0.4, 0.0]).std(ddof=0)
    assert row["sentiment_dispersion"] == pytest.approx(expected_std)
    assert row["negative_chunk_percentage"] == pytest.approx(100 * 2 / 5)


@pytest.mark.integration
def test_finbert_real_model_scores_clear_polarity_correctly():
    """One slow real-model smoke test (not mocked) to confirm the actual
    FinBERT wiring — tokenizer, pipeline, label names — works end to end."""
    scored = score_chunks(
        [
            "We are extremely pleased to report record profit and outstanding growth.",
            "Revenue collapsed and we are deeply concerned about a severe downturn.",
        ]
    )
    assert scored[0]["positive_probability"] > scored[0]["negative_probability"]
    assert scored[1]["negative_probability"] > scored[1]["positive_probability"]
