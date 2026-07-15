"""
pages/2_📊_Data_Exploration.py — EDA Dashboard
"""

import streamlit as st

st.set_page_config(page_title="Data Exploration — FakeGuard AI", page_icon="📊", layout="wide")

from collections import Counter

import pandas as pd

from components.charts import (
    class_distribution_pie,
    correlation_heatmap,
    feature_distribution,
    top_words_bar,
)
from components.styles import inject_css
from config.settings import APP_VERSION, DANGER_COLOR, PRIMARY_COLOR, SUCCESS_COLOR
from utils.helpers import load_dataset
from utils.text_processor import transform_text

inject_css()

with st.sidebar:
    st.markdown("### 📊 Data Exploration")
    st.markdown(f"*v{APP_VERSION}*")
    st.divider()
    sample_size = st.slider("Sample size for heavy charts", 1000, 10000, 5000, 500,
                             help="Reduce for faster rendering on large datasets")

st.markdown(
    """
    <div class="hero-section" style="padding:2rem;">
        <h1 style="font-size:2rem;font-weight:800;color:#F8FAFC;margin:0 0 0.5rem;">
            📊 Data Exploration
        </h1>
        <p style="color:#94A3B8;margin:0;">
            Interactive EDA of the fake reviews dataset — distributions, word clouds, correlations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

with st.spinner("Loading dataset..."):
    df = load_dataset()

df_sample = df.sample(min(sample_size, len(df)), random_state=42)

# ── Tab Layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🍩 Distribution", "📏 Text Features", "🔗 Correlation", "☁️ Word Frequency"]
)

# ── Tab 1: Distribution ────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(class_distribution_pie(df), use_container_width=True)
    with col2:
        st.markdown('<div class="section-header" style="margin-top:1rem;">📋 Class Summary</div>', unsafe_allow_html=True)
        fake_count = (df.target == 0).sum()
        real_count = (df.target == 1).sum()
        total = len(df)

        for label, count, color in [("🔴 Fake (CG)", fake_count, DANGER_COLOR), ("🟢 Real (OR)", real_count, SUCCESS_COLOR)]:
            pct = count / total
            st.markdown(
                f"""
                <div style="background:rgba(17,24,39,0.7);border:1px solid rgba(148,163,184,0.15);
                            border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.75rem;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                        <span style="color:#94A3B8;font-weight:500;">{label}</span>
                        <span style="color:{color};font-weight:700;">{count:,} ({pct:.1%})</span>
                    </div>
                    <div style="background:rgba(148,163,184,0.1);border-radius:100px;height:6px;">
                        <div style="background:{color};width:{pct*100:.1f}%;height:100%;border-radius:100px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Basic Stats</div>', unsafe_allow_html=True)
        stats = df[["char_count", "word_count"]].agg(["mean", "median", "std"]).T.round(2)
        stats.index = ["Characters per review", "Words per review"]
        st.dataframe(stats, use_container_width=True)

# ── Tab 2: Text Features ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">📏 Text Feature Distributions</div>', unsafe_allow_html=True)

    feat_options = {
        "Character count": "char_count",
        "Word count": "word_count",
        "Average word length": "avg_word_length",
        "Exclamation marks": "exclamation_count",
        "Question marks": "question_count",
        "Capital letter ratio": "capital_ratio",
    }
    selected_feat = st.selectbox("Select feature to explore", list(feat_options.keys()))
    col_name = feat_options[selected_feat]

    fig = feature_distribution(df_sample, col_name, f"{selected_feat} by Class")
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats per class
    st.markdown('<div class="section-header">📋 Statistics by Class</div>', unsafe_allow_html=True)
    stats_by_class = df.groupby("label_name")[col_name].agg(["mean", "median", "std", "min", "max"]).round(3)
    st.dataframe(stats_by_class, use_container_width=True)

# ── Tab 3: Correlation ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
    st.plotly_chart(correlation_heatmap(df), use_container_width=True)

    st.info(
        "**Reading the matrix:** Values near +1 indicate strong positive correlation, "
        "-1 strong negative. `target` = 0 (Fake) / 1 (Real)."
    )

# ── Tab 4: Word Frequency ──────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">☁️ Top Words by Class</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def compute_word_freq(texts, top_n=30):
        all_words = []
        for t in texts:
            all_words.extend(transform_text(t).split())
        return dict(Counter(all_words).most_common(top_n))

    with st.spinner("Computing word frequencies..."):
        fake_texts = df[df.target == 0]["text"].sample(min(2000, (df.target==0).sum()), random_state=42).tolist()
        real_texts = df[df.target == 1]["text"].sample(min(2000, (df.target==1).sum()), random_state=42).tolist()

        fake_freq = compute_word_freq(fake_texts)
        real_freq = compute_word_freq(real_texts)

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(top_words_bar(fake_freq, "🔴 Top Words — Fake Reviews", DANGER_COLOR), use_container_width=True)
    with col_right:
        st.plotly_chart(top_words_bar(real_freq, "🟢 Top Words — Real Reviews", SUCCESS_COLOR), use_container_width=True)

st.markdown('<div class="app-footer">FakeGuard AI · Data Exploration</div>', unsafe_allow_html=True)
