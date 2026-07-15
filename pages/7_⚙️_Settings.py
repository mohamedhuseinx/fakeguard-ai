"""
pages/7_⚙️_Settings.py — App Configuration
"""

import streamlit as st

st.set_page_config(page_title="Settings — FakeGuard AI", page_icon="⚙️", layout="wide")

from components.styles import inject_css
from config.settings import (
    APP_NAME,
    APP_VERSION,
    BEST_MODEL_NAME,
    CV_FOLDS,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)

inject_css()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(f"*v{APP_VERSION}*")

st.markdown(
    """
    <div class="hero-section" style="padding:2rem;">
        <h1 style="font-size:2rem;font-weight:800;color:#F8FAFC;margin:0 0 0.5rem;">
            ⚙️ Settings & Configuration
        </h1>
        <p style="color:#94A3B8;margin:0;">
            View and understand the application configuration parameters.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["⚙️ Model Config", "🎨 Theme", "🔧 System Info"])

# ── Tab 1: Model Config ────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">⚙️ Current Configuration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-weight:700;color:#F8FAFC;margin-bottom:1rem;">🤖 Model Parameters</div>
                <div style="display:flex;flex-direction:column;gap:0.6rem;font-size:0.9rem;">
                    <div style="display:flex;justify-content:space-between;border-bottom:1px solid rgba(148,163,184,0.1);padding-bottom:0.4rem;">
                        <span style="color:#64748B;">Best Model</span>
                        <code style="color:#6C63FF;">{BEST_MODEL_NAME}</code>
                    </div>
                    <div style="display:flex;justify-content:space-between;border-bottom:1px solid rgba(148,163,184,0.1);padding-bottom:0.4rem;">
                        <span style="color:#64748B;">Test Size</span>
                        <code style="color:#00D4AA;">{TEST_SIZE:.0%}</code>
                    </div>
                    <div style="display:flex;justify-content:space-between;border-bottom:1px solid rgba(148,163,184,0.1);padding-bottom:0.4rem;">
                        <span style="color:#64748B;">CV Folds</span>
                        <code style="color:#00D4AA;">{CV_FOLDS}</code>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#64748B;">Random State</span>
                        <code style="color:#00D4AA;">{RANDOM_STATE}</code>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-weight:700;color:#F8FAFC;margin-bottom:1rem;">🔤 TF-IDF Parameters</div>
                <div style="display:flex;flex-direction:column;gap:0.6rem;font-size:0.9rem;">
                    <div style="display:flex;justify-content:space-between;border-bottom:1px solid rgba(148,163,184,0.1);padding-bottom:0.4rem;">
                        <span style="color:#64748B;">Max Features</span>
                        <code style="color:#6C63FF;">{TFIDF_MAX_FEATURES:,}</code>
                    </div>
                    <div style="display:flex;justify-content:space-between;border-bottom:1px solid rgba(148,163,184,0.1);padding-bottom:0.4rem;">
                        <span style="color:#64748B;">N-gram Range</span>
                        <code style="color:#6C63FF;">{TFIDF_NGRAM_RANGE}</code>
                    </div>
                    <div style="display:flex;justify-content:space-between;border-bottom:1px solid rgba(148,163,184,0.1);padding-bottom:0.4rem;">
                        <span style="color:#64748B;">Min DF</span>
                        <code style="color:#00D4AA;">2</code>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#64748B;">Sublinear TF</span>
                        <code style="color:#2ED573;">True</code>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "To change these parameters, edit `config/settings.py` and re-run `python models/trainer.py`."
    )

# ── Tab 2: Theme ───────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🎨 Color Palette</div>', unsafe_allow_html=True)

    colors = [
        ("#6C63FF", "Primary", "Buttons, accents, highlights"),
        ("#00D4AA", "Secondary", "Positive indicators, gradients"),
        ("#FF4757", "Danger", "Fake labels, errors"),
        ("#2ED573", "Success", "Real labels, success states"),
        ("#FFA502", "Warning", "Medium confidence, cautions"),
        ("#0A0E1A", "Background", "Main page background"),
        ("#111827", "Card", "Card and panel backgrounds"),
    ]

    for hex_val, name, usage in colors:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:1rem;padding:0.7rem 1rem;
                        border:1px solid rgba(148,163,184,0.1);border-radius:10px;margin-bottom:0.5rem;">
                <div style="width:36px;height:36px;border-radius:8px;background:{hex_val};
                            box-shadow:0 4px 12px {hex_val}44;flex-shrink:0;"></div>
                <div>
                    <div style="color:#F8FAFC;font-weight:600;">{name}</div>
                    <div style="color:#64748B;font-size:0.8rem;">{hex_val} — {usage}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Tab 3: System Info ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🔧 System Information</div>', unsafe_allow_html=True)

    import platform
    import sys

    import sklearn
    import streamlit

    sys_info = {
        "App Name": APP_NAME,
        "App Version": APP_VERSION,
        "Python Version": sys.version.split()[0],
        "Streamlit Version": streamlit.__version__,
        "Scikit-learn Version": sklearn.__version__,
        "Platform": platform.platform(),
        "Architecture": platform.machine(),
    }

    for k, v in sys_info.items():
        st.markdown(
            f"""
            <div style="display:flex;justify-content:space-between;padding:0.5rem 1rem;
                        border-bottom:1px solid rgba(148,163,184,0.08);">
                <span style="color:#64748B;">{k}</span>
                <code style="color:#6C63FF;">{v}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="app-footer">FakeGuard AI · Settings</div>', unsafe_allow_html=True)
