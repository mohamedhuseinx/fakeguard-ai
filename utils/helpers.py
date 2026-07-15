"""
utils/helpers.py
================
General utility functions: data loading, model persistence, caching.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
import streamlit as st

from config.settings import DATASET_PATH, PIPELINE_PATH, METRICS_PATH

logger = logging.getLogger(__name__)


# ── Data Loading ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """
    Load and preprocess the fake reviews dataset with caching.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with columns: text, target, label_name.

    Raises
    ------
    FileNotFoundError
        If the dataset CSV cannot be found.
    """
    path = Path(DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)

    # Normalize column names
    if "text_" in df.columns:
        df.rename(columns={"text_": "text"}, inplace=True)

    # Encode labels
    label_map = {"CG": 0, "OR": 1}
    if "label" in df.columns:
        df["target"] = df["label"].map(label_map).fillna(0).astype(int)

    # Drop unused columns
    drop_cols = [c for c in ["category", "rating", "label"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Remove duplicates and nulls
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["text"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Add human-readable label
    df["label_name"] = df["target"].map({0: "Fake (CG)", 1: "Real (OR)"})

    # Feature engineering
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["avg_word_length"] = df["char_count"] / (df["word_count"] + 1)
    df["exclamation_count"] = df["text"].str.count("!")
    df["question_count"] = df["text"].str.count(r"\?")
    df["capital_ratio"] = df["text"].apply(
        lambda t: sum(1 for c in t if c.isupper()) / (len(t) + 1)
    )

    logger.info(f"Dataset loaded: {len(df):,} rows, {df['target'].value_counts().to_dict()}")
    return df


# ── Model Persistence ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline() -> Any:
    """
    Load the trained sklearn Pipeline from disk with Streamlit resource caching.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The trained vectorizer + classifier pipeline.

    Raises
    ------
    FileNotFoundError
        If pipeline.joblib has not been created yet.
    """
    path = Path(PIPELINE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "Model pipeline not found. Please run: python models/trainer.py"
        )
    logger.info(f"Loading pipeline from {path}")
    return joblib.load(path)


def save_pipeline(pipeline: Any) -> None:
    """
    Persist a trained sklearn Pipeline to disk using joblib compression.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline to save.
    """
    path = Path(PIPELINE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    logger.info(f"Pipeline saved to {path}")


@st.cache_data(show_spinner=False)
def load_metrics() -> Optional[dict]:
    """
    Load saved model evaluation metrics from JSON.

    Returns
    -------
    dict or None
        Metrics dict, or None if not yet generated.
    """
    path = Path(METRICS_PATH)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_metrics(metrics: dict) -> None:
    """Save model metrics to JSON for dashboard display."""
    path = Path(METRICS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")


# ── Formatting Helpers ──────────────────────────────────────────────────────────

def format_number(n: int | float) -> str:
    """Format large numbers with comma separators."""
    return f"{n:,.0f}" if isinstance(n, float) else f"{n:,}"


def confidence_color(score: float) -> str:
    """Return a color hex string based on confidence score 0–1."""
    if score >= 0.80:
        return "#2ED573"   # green
    elif score >= 0.60:
        return "#FFA502"   # orange
    else:
        return "#FF4757"   # red
