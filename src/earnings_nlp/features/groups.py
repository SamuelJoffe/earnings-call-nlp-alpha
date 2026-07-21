"""Shared per-call turn/chunk groupings used by the linguistic (Phase 7),
FinBERT (Phase 8), and change-feature (Phase 9) modules, so all three agree
on what counts as "management Q&A answers" vs. "analyst questions" etc.
"""

MANAGEMENT_ROLES = {"CEO", "CFO", "Other Management"}

GROUP_MASKS = {
    "prepared": lambda df: (df["section"] == "prepared") & df["role"].isin(MANAGEMENT_ROLES),
    "qa_management": lambda df: (df["section"] == "qa") & df["role"].isin(MANAGEMENT_ROLES),
    "analyst": lambda df: (df["section"] == "qa") & (df["role"] == "Analyst"),
    "ceo": lambda df: df["role"] == "CEO",
    "cfo": lambda df: df["role"] == "CFO",
}
