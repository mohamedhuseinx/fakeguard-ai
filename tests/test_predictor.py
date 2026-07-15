"""
tests/test_predictor.py
Unit tests for the ReviewPredictor inference engine.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PIPELINE_EXISTS = (
    Path(__file__).resolve().parent.parent / "models" / "saved" / "pipeline.joblib"
).exists()


@pytest.mark.skipif(not PIPELINE_EXISTS, reason="Pipeline not trained yet")
class TestReviewPredictor:
    """Tests for ReviewPredictor class."""

    @pytest.fixture(scope="class")
    def predictor(self):
        from models.predictor import ReviewPredictor
        return ReviewPredictor()

    def test_predictor_loads(self, predictor):
        assert predictor.pipeline is not None

    def test_predict_single_returns_dict(self, predictor):
        result = predictor.predict_single("This product is amazing!")
        assert isinstance(result, dict)

    def test_predict_single_has_required_keys(self, predictor):
        result = predictor.predict_single("Good product, works well.")
        required_keys = ["label", "label_int", "confidence", "clean_text"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_predict_single_label_is_valid(self, predictor):
        result = predictor.predict_single("Excellent quality, highly recommend!")
        assert result["label"] in ["Fake", "Real"]

    def test_predict_single_label_int_is_binary(self, predictor):
        result = predictor.predict_single("Worst product ever bought.")
        assert result["label_int"] in [0, 1]

    def test_predict_single_confidence_is_probability(self, predictor):
        result = predictor.predict_single("I love this item so much.")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_single_empty_raises(self, predictor):
        with pytest.raises(ValueError):
            predictor.predict_single("")

    def test_predict_single_whitespace_raises(self, predictor):
        with pytest.raises(ValueError):
            predictor.predict_single("   ")

    def test_predict_batch_returns_dataframe(self, predictor):
        texts = ["Great product!", "Not worth it.", "Decent quality."]
        result = predictor.predict_batch(texts)
        assert isinstance(result, pd.DataFrame)

    def test_predict_batch_correct_length(self, predictor):
        texts = ["Review one", "Review two", "Review three", "Review four"]
        result = predictor.predict_batch(texts)
        assert len(result) == len(texts)

    def test_predict_batch_has_prediction_column(self, predictor):
        result = predictor.predict_batch(["Some review text"])
        assert "prediction" in result.columns

    def test_feature_importance_returns_dataframe(self, predictor):
        fi = predictor.get_feature_importance(top_n=10)
        assert isinstance(fi, pd.DataFrame)

    def test_feature_importance_has_correct_columns(self, predictor):
        fi = predictor.get_feature_importance(top_n=5)
        if not fi.empty:
            assert "feature" in fi.columns
            assert "importance" in fi.columns
            assert "direction" in fi.columns


class TestPredictorWithoutPipeline:
    """Tests that run even without a trained pipeline."""

    def test_pipeline_missing_raises_file_not_found(self):
        """Verify helpful error when pipeline doesn't exist."""
        import tempfile
        import os
        # This test validates the error message behavior
        pipeline_path = Path("models/saved/pipeline.joblib")
        if not pipeline_path.exists():
            from utils.helpers import load_pipeline
            # Force cache clear and check error
            load_pipeline.clear()
            with pytest.raises(FileNotFoundError):
                load_pipeline()
