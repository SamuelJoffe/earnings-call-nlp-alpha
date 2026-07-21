import math

import pandas as pd
import pytest

from earnings_nlp.features.linguistic_features import (
    add_turn_features,
    aggregate_call_features,
    compute_turn_features,
)


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
