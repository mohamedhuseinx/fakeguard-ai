"""
config/settings.py
==================
Centralized configuration for the Fake Review Detection application.
All constants, paths, model parameters, and UI settings live here.
"""

import os
from pathlib import Path

# ── Project Root ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models" / "saved"
ASSETS_DIR = ROOT_DIR / "assets"
LEGACY_MODELS_DIR = MODELS_DIR / "legacy"

# ── Data ───────────────────────────────────────────────────────────────────────
DATASET_PATH = DATA_DIR / "fake reviews dataset.csv"
DATASET_SAMPLE_SIZE: int = 5000   # rows used for heavy EDA visuals (speed)

# ── Model Artifacts ────────────────────────────────────────────────────────────
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

# ── NLP Preprocessing ──────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES: int = 5000        # increased from 3000
TFIDF_NGRAM_RANGE: tuple = (1, 2)     # unigrams + bigrams
TFIDF_MIN_DF: int = 2
TFIDF_MAX_DF: float = 0.95

# ── Training ───────────────────────────────────────────────────────────────────
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42
CV_FOLDS: int = 5
BEST_MODEL_NAME: str = "SVM"

# Label mapping: CG → Computer Generated (fake), OR → Original (real)
LABEL_MAP: dict = {"CG": 0, "OR": 1}
LABEL_NAMES: dict = {0: "Fake", 1: "Real"}
CLASS_NAMES: list = ["Fake (CG)", "Real (OR)"]

# ── Model Registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict = {
    "Naive Bayes": {
        "description": "Fast probabilistic classifier, great baseline for text",
        "icon": "🎲",
    },
    "Logistic Regression": {
        "description": "Linear model with probability calibration — best performer",
        "icon": "📐",
    },
    "Decision Tree": {
        "description": "Interpretable tree-based model",
        "icon": "🌿",
    },
    "Random Forest": {
        "description": "Ensemble of 100 trees, robust to noise",
        "icon": "🌲",
    },
    "SVM": {
        "description": "Linear SVC with large margin separation",
        "icon": "⚡",
    },
}

# ── App Metadata ───────────────────────────────────────────────────────────────
APP_NAME: str = "FakeGuard AI"
APP_VERSION: str = "2.0.0"
APP_TAGLINE: str = "Advanced AI-Powered Fake Review Detection"
AUTHOR_NAME: str = "Mohamed Hussein"
AUTHOR_GITHUB: str = "https://github.com/mohamedhuseinx"
AUTHOR_LINKEDIN: str = "https://www.linkedin.com/in/mohamedhusseinx/"
AUTHOR_EMAIL: str = "mohamed.hussein.ameer@gmail.com"

# ── UI / Theme ─────────────────────────────────────────────────────────────────
PRIMARY_COLOR: str = "#6C63FF"
SECONDARY_COLOR: str = "#00D4AA"
DANGER_COLOR: str = "#FF4757"
SUCCESS_COLOR: str = "#2ED573"
WARNING_COLOR: str = "#FFA502"
BG_DARK: str = "#0A0E1A"
BG_CARD: str = "#111827"
BG_GLASS: str = "rgba(17,24,39,0.8)"
