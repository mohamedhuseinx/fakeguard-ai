"""
app.py — FakeGuard AI Main Entry Point
========================================
Premium Streamlit SaaS application for Fake Review Detection.
Run with: streamlit run app.py
"""

import logging

import streamlit as st

logging.basicConfig(level=logging.INFO)

# ── Page Configuration (must be FIRST st call) ─────────────────────────────────
st.set_page_config(
    page_title="FakeGuard AI — Fake Review Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/MOHAMED",
        "Report a bug": "https://github.com/MOHAMED/issues",
        "About": "# FakeGuard AI v2.0\nAdvanced NLP-powered fake review detection.",
    },
)

from components.styles import inject_css
from config.settings import APP_NAME, APP_TAGLINE, APP_VERSION, AUTHOR_GITHUB, AUTHOR_LINKEDIN

inject_css()

# ── Sidebar Branding ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center;padding:1rem 0 0.5rem;">
            <div style="font-size:2.5rem;margin-bottom:0.3rem;">🛡️</div>
            <div style="font-size:1.2rem;font-weight:800;color:#F8FAFC;letter-spacing:-0.02em;">{APP_NAME}</div>
            <div style="font-size:0.7rem;color:#6C63FF;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;">v{APP_VERSION}</div>
        </div>
        <hr style="border-color:rgba(148,163,184,0.1);margin:0.5rem 0 1rem;">
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Navigation**")
    st.markdown(
        """
        <div style="font-size:0.75rem;color:#64748B;margin-bottom:0.5rem;">
        Use the pages menu above ↑ to navigate
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Quick Links**")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("⭐ GitHub", AUTHOR_GITHUB, use_container_width=True)
    with col2:
        st.link_button("💼 LinkedIn", AUTHOR_LINKEDIN, use_container_width=True)

    st.markdown(
        f"""
        <div style="position:absolute;bottom:1rem;left:0;right:0;text-align:center;
                    font-size:0.7rem;color:#334155;padding:0 1rem;">
            {APP_NAME} © 2024 · MIT License
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Main Page (Landing / Home redirect) ────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero-section" style="margin-bottom:2rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">🛡️</div>
        <h1 style="font-size:3rem;font-weight:900;color:#F8FAFC;margin:0 0 0.5rem;
                   background:linear-gradient(135deg,#6C63FF,#00D4AA);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            {APP_NAME}
        </h1>
        <p style="font-size:1.2rem;color:#94A3B8;max-width:600px;margin:0 auto 1.5rem;">
            {APP_TAGLINE}
        </p>
        <div style="display:flex;gap:0.75rem;justify-content:center;flex-wrap:wrap;">
            <span class="stat-badge">🤖 5 ML Models</span>
            <span class="stat-badge">📊 40,412 Reviews</span>
            <span class="stat-badge">⚡ AUC 0.963</span>
            <span class="stat-badge">🎯 89.7% F1 Score</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI Cards ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Best F1 Score", "89.7%", "↑ vs baseline")
c2.metric("📈 AUC-ROC", "0.963", "SVM model")
c3.metric("📦 Dataset Size", "40,412", "balanced 50/50")
c4.metric("🏆 Best Model", "SVM", "Linear kernel")

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature Grid ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="section-header">✨ Application Features</div>
    """,
    unsafe_allow_html=True,
)

features = [
    ("📊", "Data Exploration", "Interactive EDA with word clouds, distributions, and correlation heatmaps"),
    ("📈", "Analytics Dashboard", "Executive KPIs, class imbalance charts, feature engineering insights"),
    ("🤖", "AI Prediction", "Single review and CSV batch prediction with confidence scores"),
    ("📉", "Model Performance", "Full metrics: Confusion Matrix, ROC, PR Curve, Classification Report"),
    ("🔍", "Explainability", "SHAP values and TF-IDF feature importance for transparent AI"),
    ("📂", "Dataset Browser", "Searchable, filterable view of the full dataset with download"),
]

cols = st.columns(3)
for i, (icon, title, desc) in enumerate(features):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align:center;padding:1.5rem;">
                <div style="font-size:2rem;margin-bottom:0.75rem;">{icon}</div>
                <div style="font-weight:700;color:#F8FAFC;margin-bottom:0.5rem;font-size:1rem;">{title}</div>
                <div style="color:#64748B;font-size:0.85rem;line-height:1.5;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="app-footer">
        Built with ❤️ using <strong>Python</strong> · <strong>Scikit-learn</strong> · <strong>Streamlit</strong> · <strong>Plotly</strong>
        &nbsp;|&nbsp; <a href="{AUTHOR_GITHUB}">GitHub</a>
        &nbsp;|&nbsp; <a href="{AUTHOR_LINKEDIN}">LinkedIn</a>
        &nbsp;|&nbsp; v{APP_VERSION}
    </div>
    """,
    unsafe_allow_html=True,
)
