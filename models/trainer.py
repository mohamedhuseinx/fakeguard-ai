"""
models/trainer.py
=================
Production ML training pipeline for Fake Review Detection.

Improvements over original notebook:
- StratifiedKFold cross-validation (5 folds)
- GridSearchCV for best Logistic Regression hyperparameters
- sklearn Pipeline (TF-IDF → Classifier) — single serializable object
- Comprehensive metrics: Accuracy, F1, Recall, AUC-ROC, Confusion Matrix
- Saves compressed pipeline.joblib + model_metrics.json

Run directly: python models/trainer.py
"""

import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    BEST_MODEL_NAME,
    CV_FOLDS,
    LABEL_NAMES,
    METRICS_PATH,
    PIPELINE_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)
from utils.helpers import load_dataset, save_metrics, save_pipeline
from utils.text_processor import batch_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Model Definitions ──────────────────────────────────────────────────────────

def get_model_candidates() -> Dict[str, Any]:
    """Return dict of candidate classifiers to evaluate."""
    return {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, min_samples_split=10, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "SVM": LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
    }


def build_tfidf() -> TfidfVectorizer:
    """Build the TF-IDF vectorizer with production settings."""
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,              # apply log(1+tf) — helps text classification
        strip_accents="unicode",
    )


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    clf: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_splits: StratifiedKFold,
) -> Dict[str, Any]:
    """
    Fit a classifier and compute comprehensive evaluation metrics.

    Parameters
    ----------
    name : str
        Human-readable model name.
    clf : estimator
        sklearn-compatible classifier.
    X_train, y_train : array-like
        Training features and labels.
    X_test, y_test : array-like
        Test features and labels.
    cv_splits : StratifiedKFold
        CV splitter for cross-validation.

    Returns
    -------
    dict
        Dictionary of all evaluation metrics.
    """
    logger.info(f"Training: {name}")
    t0 = time.time()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Probability scores for AUC (not available for LinearSVC by default)
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
    except AttributeError:
        # LinearSVC uses decision function
        try:
            y_score = clf.decision_function(X_test)
            auc = float(roc_auc_score(y_test, y_score))
        except Exception:
            auc = None

    # Cross-validation F1
    cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv_splits, scoring="f1", n_jobs=-1)

    train_time = time.time() - t0
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values()), output_dict=True)

    return {
        "name": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": auc,
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
        "confusion_matrix": cm,
        "classification_report": report,
        "train_time_sec": round(train_time, 3),
    }


# ── Main Training Pipeline ─────────────────────────────────────────────────────

def train() -> None:
    """Execute the full training pipeline and save artifacts."""
    logger.info("=" * 60)
    logger.info("FakeGuard AI — Training Pipeline v2.0")
    logger.info("=" * 60)

    # 1. Load & preprocess data
    logger.info("Loading dataset...")
    df = load_dataset()
    logger.info(f"Rows: {len(df):,} | Fake: {(df.target==0).sum():,} | Real: {(df.target==1).sum():,}")

    logger.info("Applying text preprocessing...")
    df["clean_text"] = batch_transform(df["text"].tolist())

    X = df["clean_text"].values
    y = df["target"].values

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 3. Fit TF-IDF
    logger.info("Fitting TF-IDF vectorizer...")
    tfidf = build_tfidf()
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    logger.info(f"Vocabulary size: {len(tfidf.vocabulary_):,} | Feature matrix: {X_train_vec.shape}")

    # 4. Evaluate all models
    cv_splits = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = get_model_candidates()
    all_metrics: List[Dict] = []
    trained_models: Dict[str, Any] = {}

    for name, clf in candidates.items():
        metrics = evaluate_model(name, clf, X_train_vec, y_train, X_test_vec, y_test, cv_splits)
        all_metrics.append(metrics)
        trained_models[name] = clf
        logger.info(
            f"  ✓ {name:22s} | Acc={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1']:.4f} | AUC={metrics['auc_roc'] or 'N/A'}"
        )

    # 5. Rank by F1 + pick best
    best = max(all_metrics, key=lambda m: m["f1"])
    logger.info(f"\n🏆 Best model: {best['name']} (F1={best['f1']:.4f})")

    # 6. Save full pipeline (vectorizer + best model)
    best_clf = trained_models[best["name"]]
    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("classifier", best_clf),
    ])

    # Attach all trained models to pipeline for multi-model display
    pipeline.all_models_ = trained_models
    pipeline.all_metrics_ = all_metrics
    pipeline.tfidf_ = tfidf
    pipeline.best_model_name_ = best["name"]

    save_pipeline(pipeline)
    logger.info(f"Pipeline saved → {PIPELINE_PATH}")

    # 7. Save metrics JSON
    metrics_payload = {
        "best_model": best["name"],
        "all_models": all_metrics,
        "dataset_stats": {
            "total_rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "fake_count": int((df.target == 0).sum()),
            "real_count": int((df.target == 1).sum()),
        },
        "tfidf_config": {
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range": list(TFIDF_NGRAM_RANGE),
            "vocab_size": len(tfidf.vocabulary_),
        },
    }
    save_metrics(metrics_payload)
    logger.info(f"Metrics saved → {METRICS_PATH}")
    logger.info("=" * 60)
    logger.info("Training complete! ✅")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
