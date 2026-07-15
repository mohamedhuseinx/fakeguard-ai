"""
tests/test_text_processor.py
Unit tests for the NLP text preprocessing pipeline.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.text_processor import batch_transform, transform_text


class TestTransformText:
    """Tests for transform_text function."""

    def test_basic_lowercase(self):
        result = transform_text("Hello World")
        assert result == result.lower()

    def test_removes_stopwords(self):
        result = transform_text("this is a very good product")
        # 'this', 'is', 'a', 'very' should be removed as stopwords
        assert "is" not in result.split()

    def test_removes_punctuation(self):
        result = transform_text("Great product!!! Loved it.")
        assert "!" not in result
        assert "." not in result

    def test_removes_numbers(self):
        result = transform_text("I bought 3 items at 100 dollars")
        tokens = result.split()
        assert not any(tok.isdigit() for tok in tokens)

    def test_removes_urls(self):
        result = transform_text("Visit https://example.com for more")
        assert "https" not in result
        assert "example" not in result.lower()

    def test_stemming_applied(self):
        # "running" → "run" after stemming
        result1 = transform_text("running")
        result2 = transform_text("run")
        assert result1 == result2

    def test_empty_string(self):
        assert transform_text("") == ""

    def test_whitespace_only(self):
        assert transform_text("   ") == ""

    def test_non_string_input(self):
        # Should handle gracefully
        result = transform_text(None)
        assert result == ""

    def test_returns_string(self):
        result = transform_text("This is a test review.")
        assert isinstance(result, str)

    def test_realistic_fake_review(self):
        text = "This product is absolutely amazing!!! Best purchase ever! 5 stars!!!"
        result = transform_text(text)
        # Should produce non-empty tokens
        assert len(result.strip()) > 0

    def test_realistic_real_review(self):
        text = "I've been using this for 3 weeks. Battery life is shorter than advertised."
        result = transform_text(text)
        assert isinstance(result, str)


class TestBatchTransform:
    """Tests for batch_transform function."""

    def test_returns_list(self):
        texts = ["Hello world", "Another review"]
        result = batch_transform(texts)
        assert isinstance(result, list)

    def test_same_length(self):
        texts = ["review one", "review two", "review three"]
        result = batch_transform(texts)
        assert len(result) == len(texts)

    def test_empty_list(self):
        result = batch_transform([])
        assert result == []

    def test_consistent_with_single(self):
        text = "This is a test review!"
        single = transform_text(text)
        batch = batch_transform([text])
        assert batch[0] == single
