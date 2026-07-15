"""
models/predictor.py
===================
Inference engine for fake review detection.
Supports single prediction, batch prediction, SHAP explanations,
and probability confidence scores.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import LABEL_NAMES
from utils.helpers import load_pipeline
from utils.text_processor import batch_transform, transform_text

logger = logging.getLogger(__name__)


class ReviewPredictor:
    """
    High-level inference class for the Fake Review Detection pipeline.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Loaded trained pipeline (TF-IDF + classifier).
    """

    def __init__(self) -> None:
        self.pipeline = load_pipeline()

    # ── Single Prediction ───────────────────────────────────────────────────────

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict whether a single review is fake or real.

        Parameters
        ----------
        text : str
            Raw review text.

        Returns
        -------
        dict with keys:
            label       : "Fake" or "Real"
            label_int   : 0 (Fake) or 1 (Real)
            confidence  : float 0–1
            probabilities: dict {"Fake": float, "Real": float}
            clean_text  : preprocessed text
            all_model_votes: dict of per-model predictions
        """
        if not text or not text.strip():
            raise ValueError("Review text cannot be empty.")

        clean = transform_text(text)
        result = self._predict_from_clean([clean])[0]
        result["clean_text"] = clean
        result["all_model_votes"] = self._all_model_votes(clean)
        return result

    def predict_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Predict a list of reviews and return a DataFrame.

        Parameters
        ----------
        texts : List[str]
            List of raw review texts.

        Returns
        -------
        pd.DataFrame
            Columns: text, clean_text, prediction, confidence, label
        """
        clean_texts = batch_transform(texts)
        predictions = self._predict_from_clean(clean_texts)

        records = []
        for i, (raw, pred) in enumerate(zip(texts, predictions)):
            records.append({
                "text": raw[:120] + ("..." if len(raw) > 120 else ""),
                "clean_text": clean_texts[i],
                "prediction": pred["label"],
                "confidence": f"{pred['confidence']:.1%}",
                "confidence_raw": pred["confidence"],
                "label_int": pred["label_int"],
            })
        return pd.DataFrame(records)

    # ── Multi-Model Voting ──────────────────────────────────────────────────────

    def _all_model_votes(self, clean_text: str) -> Dict[str, str]:
        """
        Get individual predictions from all trained models (majority vote display).

        Parameters
        ----------
        clean_text : str
            Preprocessed text string.

        Returns
        -------
        dict
            {model_name: "Fake" | "Real"}
        """
        tfidf = self.pipeline.tfidf_
        all_models = getattr(self.pipeline, "all_models_", {})

        if not all_models:
            return {}

        vec = tfidf.transform([clean_text])
        votes = {}
        for name, clf in all_models.items():
            try:
                pred_int = int(clf.predict(vec)[0])
                votes[name] = LABEL_NAMES.get(pred_int, "Unknown")
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
                votes[name] = "Error"
        return votes

    # ── Internal Helpers ────────────────────────────────────────────────────────

    def _predict_from_clean(self, clean_texts: List[str]) -> List[Dict[str, Any]]:
        """Run the main pipeline on a list of preprocessed texts."""
        clf = self.pipeline.named_steps["classifier"]
        tfidf = self.pipeline.named_steps["tfidf"]

        vecs = tfidf.transform(clean_texts)
        predictions = clf.predict(vecs)

        results = []
        for i, pred_int in enumerate(predictions):
            # Get probabilities if available
            try:
                proba = clf.predict_proba(vecs[i])[0]
                confidence = float(proba[pred_int])
                probs = {
                    "Fake": float(proba[0]),
                    "Real": float(proba[1]),
                }
            except AttributeError:
                # LinearSVC → use decision function as proxy
                try:
                    score = float(clf.decision_function(vecs[i])[0])
                    # sigmoid transform for approximate probability
                    prob_real = 1 / (1 + np.exp(-score))
                    prob_fake = 1 - prob_real
                    confidence = prob_real if pred_int == 1 else prob_fake
                    probs = {"Fake": prob_fake, "Real": prob_real}
                except Exception:
                    confidence = 1.0
                    probs = {}

            results.append({
                "label": LABEL_NAMES.get(int(pred_int), "Unknown"),
                "label_int": int(pred_int),
                "confidence": confidence,
                "probabilities": probs,
            })
        return results

    # ── SHAP Explanations ───────────────────────────────────────────────────────

    def explain_shap(self, text: str, max_features: int = 15) -> Optional[Dict]:
        """
        Generate SHAP feature importance for a single prediction.

        Parameters
        ----------
        text : str
            Raw review text.
        max_features : int
            Number of top contributing features to return.

        Returns
        -------
        dict or None
            {feature_names: [...], shap_values: [...], base_value: float}
            Returns None if SHAP is unavailable.
        """
        try:
            import shap

            clean = transform_text(text)
            tfidf = self.pipeline.named_steps["tfidf"]
            clf = self.pipeline.named_steps["classifier"]

            vec = tfidf.transform([clean])
            feature_names = np.array(tfidf.get_feature_names_out())

            # Use LinearExplainer for linear models
            explainer = shap.LinearExplainer(clf, shap.maskers.Independent(vec))
            shap_values = explainer.shap_values(vec)

            if isinstance(shap_values, list):
                sv = shap_values[1][0]  # class 1 (Real)
            else:
                sv = shap_values[0]

            # Get top features
            abs_sv = np.abs(sv)
            top_idx = np.argsort(abs_sv)[::-1][:max_features]

            return {
                "feature_names": feature_names[top_idx].tolist(),
                "shap_values": sv[top_idx].tolist(),
                "base_value": float(explainer.expected_value
                                    if not isinstance(explainer.expected_value, np.ndarray)
                                    else explainer.expected_value[1]),
            }
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return None

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Extract TF-IDF feature importance from the trained classifier.

        Works for Logistic Regression and LinearSVC (coefficient-based).

        Parameters
        ----------
        top_n : int
            Number of top features per class to return.

        Returns
        -------
        pd.DataFrame
            Columns: feature, importance, direction
        """
        clf = self.pipeline.named_steps["classifier"]
        tfidf = self.pipeline.named_steps["tfidf"]
        feature_names = np.array(tfidf.get_feature_names_out())

        try:
            coefs = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_importances_
        except AttributeError:
            return pd.DataFrame()

        # Top fake-leaning and real-leaning features
        top_real_idx = np.argsort(coefs)[::-1][:top_n]
        top_fake_idx = np.argsort(coefs)[:top_n]

        rows = []
        for idx in top_real_idx:
            rows.append({"feature": feature_names[idx], "importance": coefs[idx], "direction": "Real"})
        for idx in top_fake_idx:
            rows.append({"feature": feature_names[idx], "importance": abs(coefs[idx]), "direction": "Fake"})

        return pd.DataFrame(rows)
