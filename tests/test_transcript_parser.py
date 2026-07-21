import json

import pytest

from earnings_nlp.processing.clean_transcripts import parse_transcript, parse_transcript_file

SAMPLE_TRANSCRIPT = {
    "symbol": "TEST",
    "quarter": "2024Q1",
    "transcript": [
        {
            "speaker": "Operator",
            "title": "Operator",
            "content": "Good day, and thank you for standing by. Welcome to the Q1 2024 earnings call.",
        },
        {
            "speaker": "Jane Doe",
            "title": "Chairman and Chief Executive Officer",
            "content": "Thank you, operator. We are pleased to report strong revenue growth this quarter, "
            "driven by continued momentum across our core product lines.",
        },
        {
            "speaker": "John Smith",
            "title": "Chief Financial Officer",
            "content": "Thanks, Jane. Turning to the numbers, gross margin expanded year over year "
            "and free cash flow remained healthy.",
        },
        {
            "speaker": "Operator",
            "title": "Operator",
            "content": "Page 3 of 12",
        },
        {
            "speaker": "Operator",
            "title": "Operator",
            "content": "We will now begin the question-and-answer session. Your first question comes from the line of Alex Analyst.",
        },
        {
            "speaker": "Alex Analyst",
            "title": "Analyst, Big Bank Securities",
            "content": "Thanks for taking my question. Can you comment on the sustainability of that margin expansion?",
        },
        {
            "speaker": "John Smith",
            "title": "Chief Financial Officer",
            "content": "Sure. We expect the majority of the improvement to persist, though there is some seasonality.",
        },
        {
            "speaker": "John Smith",
            "title": "Chief Financial Officer",
            "content": "Sure. We expect the majority of the improvement to persist, though there is some seasonality.",
        },
        {
            "speaker": "Operator",
            "title": "Operator",
            "content": "Thank you.",
        },
        {
            "speaker": "Jane Doe",
            "title": "Chairman and Chief Executive Officer",
            "content": "Thanks.",
        },
        {
            "speaker": "",
            "title": "",
            "content": "",
        },
    ],
}


@pytest.fixture
def parsed_df():
    return parse_transcript("TEST", "2024Q1", SAMPLE_TRANSCRIPT)


def test_ceo_is_identified(parsed_df):
    ceo_rows = parsed_df[parsed_df["speaker"] == "Jane Doe"]
    assert (ceo_rows["role"] == "CEO").all()


def test_cfo_is_identified(parsed_df):
    cfo_rows = parsed_df[parsed_df["speaker"] == "John Smith"]
    assert (cfo_rows["role"] == "CFO").all()


def test_analysts_are_not_management(parsed_df):
    analyst_rows = parsed_df[parsed_df["speaker"] == "Alex Analyst"]
    assert (analyst_rows["role"] == "Analyst").all()
    assert not analyst_rows["role"].isin(["CEO", "CFO", "Other Management"]).any()


def test_operator_turns_are_removed(parsed_df):
    assert not (parsed_df["role"] == "Operator").any()


def test_qa_starts_after_operator_announcement(parsed_df):
    prepared = parsed_df[parsed_df["section"] == "prepared"]
    qa = parsed_df[parsed_df["section"] == "qa"]
    assert len(prepared) > 0
    assert len(qa) > 0
    # the CEO's opening remark must be prepared, the analyst's question must be qa
    assert prepared[prepared["speaker"] == "Jane Doe"]["text"].str.contains("revenue growth").any()
    assert qa[qa["speaker"] == "Alex Analyst"]["text"].str.contains("margin expansion").any()


def test_duplicate_turns_are_removed(parsed_df):
    matching = parsed_df[parsed_df["text"].str.contains("majority of the improvement")]
    assert len(matching) == 1


def test_page_artifacts_and_empty_turns_are_removed(parsed_df):
    assert not parsed_df["text"].str.contains("Page 3 of 12").any()
    assert (parsed_df["text"].str.len() > 0).all()


def test_short_non_substantive_remarks_are_removed(parsed_df):
    assert not (parsed_df["text"] == "Thanks.").any()


def test_expected_columns(parsed_df):
    assert list(parsed_df.columns) == [
        "ticker",
        "quarter",
        "speaker",
        "title",
        "role",
        "section",
        "text",
    ]
    assert (parsed_df["ticker"] == "TEST").all()
    assert (parsed_df["quarter"] == "2024Q1").all()


def test_parse_transcript_file_infers_ticker_and_quarter(tmp_path):
    file_path = tmp_path / "AAPL_2024Q1.json"
    file_path.write_text(json.dumps(SAMPLE_TRANSCRIPT))

    df = parse_transcript_file(file_path)

    assert (df["ticker"] == "AAPL").all()
    assert (df["quarter"] == "2024Q1").all()
