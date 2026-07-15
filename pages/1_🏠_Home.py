"""
pages/1_🏠_Home.py — Dashboard Home Page
"""

import streamlit as st

st.set_page_config(page_title="Home — FakeGuard AI", page_icon="🏠", layout="wide")

from components.styles import inject_css
from config.settings import APP_NAME, APP_TAGLINE, APP_VERSION
from utils.helpers import load_dataset, load_metrics

inject_css()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 🛡️ {APP_NAME}")
    st.markdown(f"*v{APP_VERSION}*")
    st.divider()
    st.info("**Tip:** Use the sidebar pages to navigate through the full dashboard.")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero-section">
        <h1 style="font-size:2.5rem;font-weight:900;margin:0 0 0.75rem;
                   background:linear-gradient(135deg,#6C63FF,#00D4AA);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🛡️ {APP_NAME} Dashboard
        </h1>
        <p style="color:#94A3B8;font-size:1.05rem;max-width:600px;margin:0 auto;">
            {APP_TAGLINE}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Load data and metrics ───────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    try:
        df = load_dataset()
        metrics = load_metrics()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# ── KPI Row 1 ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
best = None
if metrics:
    best = max(metrics["all_models"], key=lambda m: m["f1"])

with c1:
    st.metric("📦 Total Reviews", f"{len(df):,}", "dataset size")
with c2:
    st.metric("🔴 Fake Reviews", f"{(df.target==0).sum():,}", f"{(df.target==0).mean():.1%}")
with c3:
    st.metric("🟢 Real Reviews", f"{(df.target==1).sum():,}", f"{(df.target==1).mean():.1%}")
with c4:
    if best:
        st.metric("🏆 Best F1", f"{best['f1']:.4f}", best["name"])
    else:
        st.metric("🏆 Best F1", "—")
with c5:
    if best:
        auc = best.get("auc_roc") or 0
        st.metric("📈 Best AUC", f"{auc:.4f}", best["name"])
    else:
        st.metric("📈 Best AUC", "—")

st.markdown("<br>", unsafe_allow_html=True)

# ── Model Leaderboard ──────────────────────────────────────────────────────────
if metrics:
    st.markdown('<div class="section-header">🏅 Model Leaderboard</div>', unsafe_allow_html=True)

    import pandas as pd
    leaderboard = pd.DataFrame(metrics["all_models"])[
        ["name", "accuracy", "precision", "recall", "f1", "auc_roc", "cv_f1_mean", "train_time_sec"]
    ].sort_values("f1", ascending=False).reset_index(drop=True)
    leaderboard.index = leaderboard.index + 1

    leaderboard.columns = [
        "Model", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "CV F1 (Mean)", "Train Time (s)"
    ]
    for col in ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "CV F1 (Mean)"]:
        leaderboard[col] = leaderboard[col].apply(
            lambda x: f"{x:.4f}" if x is not None else "N/A"
        )
    leaderboard["Train Time (s)"] = leaderboard["Train Time (s)"].apply(lambda x: f"{x:.2f}s")

    # Add rank medal
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    leaderboard.insert(0, "Rank", medals[:len(leaderboard)])

    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset Stats ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Dataset Statistics</div>', unsafe_allow_html=True)

    ds = metrics.get("dataset_stats", {})
    tfidf_cfg = metrics.get("tfidf_config", {})

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-weight:700;color:#F8FAFC;margin-bottom:1rem;font-size:1rem;">📦 Dataset</div>
                <div style="display:flex;flex-direction:column;gap:0.5rem;">
                    <div style="display:flex;justify-content:space-between;color:#94A3B8;">
                        <span>Total rows</span><strong style="color:#F8FAFC">{ds.get('total_rows',len(df)):,}</strong>
                    </div>
                    <div style="display:flex;justify-content:space-between;color:#94A3B8;">
                        <span>Training samples</span><strong style="color:#F8FAFC">{ds.get('train_rows','—'):,}</strong>
                    </div>
                    <div style="display:flex;justify-content:space-between;color:#94A3B8;">
                        <span>Test samples</span><strong style="color:#F8FAFC">{ds.get('test_rows','—'):,}</strong>
                    </div>
                    <div style="display:flex;justify-content:space-between;color:#94A3B8;">
                        <span>Fake (CG)</span><strong style="color:#FF4757">{ds.get('fake_count','—'):,}</strong>
                    </div>
                    <div style="display:flex;justify-content:space-between;color:#94A3B8;">
                        <span>Real (OR)</span><strong style="color:#2ED573">{ds.get('real_count','—'):,}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-weight:700;color:#F8FAFC;margin-bottom:1rem;font-size:1rem;">⚙️ TF-IDF Config</div>
                <div style="display:flex;flex-direction:column;gap:0.5rem;">
                    <div style="display:flex;justify-content:space-between;color:#94A3B8;">
                        <span>Max features</span><strong style="color:#F8FAFC">{tfidf_cfg.get('max_features',5000):,}</strong>
                    </div>
                    <div style="display:flex;justify-content:space-between;color:#94A3B8;">
                        <span>N-gram range</span><strong style="color:#F8FAFC">{tfidf_cfg.get('ngram_range',[1,2])}</strong>
                    </div>
                    <div style="display:flex;justify-content:space-between;color:#94A3B8;">
                        <span>Vocabulary size</span><strong style="color:#F8FAFC">{tfidf_cfg.get('vocab_size',5000):,}</strong>
                    </div>
                    <div style="display:flex;justify-content:space-between;color:#94A3B8;">
                        <span>Sublinear TF</span><strong style="color:#6C63FF">✓ Enabled</strong>
                    </div>
                    <div style="display:flex;justify-content:space-between;color:#94A3B8;">
                        <span>Best model</span><strong style="color:#00D4AA">{metrics.get('best_model','SVM')}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown(
    """<div class="app-footer">FakeGuard AI · Built with Python, Scikit-learn & Streamlit</div>""",
    unsafe_allow_html=True,
)
