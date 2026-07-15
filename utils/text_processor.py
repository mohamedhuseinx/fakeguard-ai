"""
utils/text_processor.py
=======================
Optimized NLP text preprocessing pipeline for fake review detection.
Fixes the O(n²) stopwords bug from the original notebook.
"""

import logging
import re
import string
from functools import lru_cache
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ── One-time NLTK downloads ────────────────────────────────────────────────────
def ensure_nltk_resources() -> None:
    """Download required NLTK resources if not already present."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)


ensure_nltk_resources()

# ── Pre-load expensive objects once (O(1) lookups) ─────────────────────────────
_STEMMER = PorterStemmer()
_STOP_WORDS: frozenset = frozenset(stopwords.words("english"))  # set → O(1)
_PUNCT_SET: frozenset = frozenset(string.punctuation)


@lru_cache(maxsize=10_000)
def _stem(word: str) -> str:
    """Cached stem lookup — avoids re-stemming identical tokens."""
    return _STEMMER.stem(word)


def transform_text(text: str) -> str:
    """
    Clean, tokenize, remove stopwords/punctuation, and stem input text.

    Performance improvements over original:
    - Stopwords stored in a frozenset (O(1) vs O(n) list lookup)
    - Stemming results cached via lru_cache
    - Single-pass filtering instead of three separate loops

    Parameters
    ----------
    text : str
        Raw review text to preprocess.

    Returns
    -------
    str
        Space-joined preprocessed tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase + basic cleanup
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"\d+", " ", text)               # remove digits

    # 2. Tokenize
    tokens: List[str] = word_tokenize(text)

    # 3. Single-pass: keep alphanumeric, drop stopwords & punctuation, stem
    processed = [
        _stem(tok)
        for tok in tokens
        if tok.isalnum()
        and tok not in _STOP_WORDS
        and tok not in _PUNCT_SET
    ]

    return " ".join(processed)


def batch_transform(texts: List[str]) -> List[str]:
    """
    Apply transform_text to a list of review strings.

    Parameters
    ----------
    texts : List[str]
        List of raw review texts.

    Returns
    -------
    List[str]
        List of preprocessed texts.
    """
    return [transform_text(t) for t in texts]
