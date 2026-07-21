"""Split turn text into <=512-token chunks for FinBERT.

FinBERT (ProsusAI/finbert) has a maximum position length of 512 tokens
including the [CLS]/[SEP] special tokens, so a turn's raw text must be
split on token boundaries (not just word/character counts) before it is
sent through the classifier. The tokenizer and classifier pipeline are
loaded lazily and cached at module level, since loading the model is slow
and only needed once per process.
"""

from __future__ import annotations

from functools import lru_cache

MODEL_NAME = "ProsusAI/finbert"
MAX_CHUNK_TOKENS = 510  # 512 minus room for [CLS] and [SEP]


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME)


@lru_cache(maxsize=1)
def get_classifier():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=get_tokenizer(),
        truncation=True,
        top_k=None,
    )


def chunk_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:
    """Split `text` into a list of chunks, each within `max_tokens` FinBERT
    tokens. Returns an empty list for empty/whitespace-only text.
    """
    text = (text or "").strip()
    if not text:
        return []

    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks = [token_ids[i : i + max_tokens] for i in range(0, len(token_ids), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
