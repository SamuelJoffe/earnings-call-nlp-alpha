"""Classify each transcript turn's speaker role and transcript section.

Alpha Vantage's EARNINGS_CALL_TRANSCRIPT response gives a `title` string per
turn (e.g. "Chief Executive Officer", "Analyst") but no explicit prepared/QA
boundary, so the section is inferred from the turn sequence.
"""

from __future__ import annotations

import re

ROLE_CEO = "CEO"
ROLE_CFO = "CFO"
ROLE_OTHER_MANAGEMENT = "Other Management"
ROLE_ANALYST = "Analyst"
ROLE_OPERATOR = "Operator"
ROLE_UNKNOWN = "Unknown"

_CEO_PATTERN = re.compile(r"\bchief executive\b|\bceo\b", re.IGNORECASE)
_CFO_PATTERN = re.compile(r"\bchief financial\b|\bcfo\b", re.IGNORECASE)
_ANALYST_PATTERN = re.compile(r"\banalyst\b", re.IGNORECASE)
_OPERATOR_PATTERN = re.compile(r"\boperator\b", re.IGNORECASE)
_MANAGEMENT_PATTERN = re.compile(
    r"\bchief\b|\bpresident\b|\bvice president\b|\bvp\b|\bhead of\b|"
    r"\bdirector\b|\bchairman\b|\btreasurer\b|\bcontroller\b",
    re.IGNORECASE,
)

_QA_START_PATTERNS = re.compile(
    r"first question|"
    r"we('ll| will)? now (begin|open|conduct)\b.{0,40}question|"
    r"now open (the (call|line)|it up) for questions",
    re.IGNORECASE,
)


def classify_role(speaker: str, title: str) -> str:
    """Classify a single turn's speaker into a coarse role bucket."""
    speaker = speaker or ""
    title = title or ""
    combined = f"{speaker} {title}"

    if _OPERATOR_PATTERN.search(combined):
        return ROLE_OPERATOR
    if _CEO_PATTERN.search(title):
        return ROLE_CEO
    if _CFO_PATTERN.search(title):
        return ROLE_CFO
    if _ANALYST_PATTERN.search(title):
        return ROLE_ANALYST
    if _MANAGEMENT_PATTERN.search(title):
        return ROLE_OTHER_MANAGEMENT
    return ROLE_UNKNOWN


def label_sections(turns: list[dict], content_field: str = "content") -> list[str]:
    """Label each turn as 'prepared' or 'qa'.

    The Q&A portion is assumed to start at the first Operator turn that
    announces the question-and-answer session; if no such announcement is
    found, it falls back to the first turn spoken by an Analyst.
    """
    qa_start_index = None

    for i, turn in enumerate(turns):
        role = classify_role(turn.get("speaker", ""), turn.get("title", ""))
        text = turn.get(content_field, "") or ""
        if role == ROLE_OPERATOR and _QA_START_PATTERNS.search(text):
            qa_start_index = i + 1
            break

    if qa_start_index is None:
        for i, turn in enumerate(turns):
            role = classify_role(turn.get("speaker", ""), turn.get("title", ""))
            if role == ROLE_ANALYST:
                qa_start_index = i
                break

    if qa_start_index is None:
        qa_start_index = len(turns)

    return [
        "prepared" if i < qa_start_index else "qa"
        for i in range(len(turns))
    ]
