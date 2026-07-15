"""
pages/3_📈_Analytics.py — Executive Analytics Dashboard
"""

import streamlit as st

st.set_page_config(page_title="Analytics — FakeGuard AI", page_icon="📈", layout="wide")

import plotly.express as px
import plotly.graph_objects as go

from components.charts import feature_distribution, model_comparison_bar
from components.styles import inject_css
from config.settings import APP_VERSION, DANGER_COLOR, PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR
from utils.helpers import load_dataset, load_metrics

inject_css()

with st.sidebar:
    st.markdown("### 📈 Analytics")
    st.markdown(f"*v{APP_VERSION}*")
    st.divider()
    st.info("**Analytics** shows model comparison and feature engineering insights.")

st.markdown(
    """
    <div class="hero-section" style="padding:2rem;">
        <h1 style="font-size:2rem;font-weight:800;color:#F8FAFC;margin:0 0 0.5rem;">
            📈 Analytics Dashboard
        </h1>
        <p style="color:#94A3B8;margin:0;">
            Model performance comparison, feature engineering, and business insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

with st.spinner("Loading..."):
    df = load_dataset()
    metrics = load_metrics()

if not metrics:
    st.warning("No metrics found. Run `python models/trainer.py` first.")
    st.stop()

all_models = metrics["all_models"]
best_name = metrics["best_model"]

# ── Tab Layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏅 Model Comparison", "🔬 Feature Engineering", "📊 Business Insights"])

# ── Tab 1: Model Comparison ────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">🏅 Algorithm Showdown</div>', unsafe_allow_html=True)
    st.plotly_chart(model_comparison_bar(all_models), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⏱️ Training Time vs F1 Score</div>', unsafe_allow_html=True)

    import pandas as pd
    df_m = pd.DataFrame(all_models)

    fig_scatter = go.Figure()
    colors_map = {
        "Naive Bayes": PRIMARY_COLOR,
        "Logistic Regression": SECONDARY_COLOR,
        "Decision Tree": "#FFA502",
        "Random Forest": SUCCESS_COLOR,
        "SVM": DANGER_COLOR,
    }
    for _, row in df_m.iterrows():
        fig_scatter.add_trace(go.Scatter(
            x=[row["train_time_sec"]],
            y=[row["f1"]],
            mode="markers+text",
            marker=dict(size=18, color=colors_map.get(row["name"], PRIMARY_COLOR),
                        line=dict(width=2, color="white")),
            text=[row["name"]],
            textposition="top center",
            textfont=dict(color="#F8FAFC", size=11),
            name=row["name"],
            hovertemplate=f"<b>{row['name']}</b><br>Train time: %{{x:.2f}}s<br>F1: %{{y:.4f}}<extra></extra>",
        ))

    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94A3B8"),
        xaxis=dict(title="Training Time (seconds)", gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(title="F1 Score", gridcolor="rgba(148,163,184,0.1)"),
        showlegend=False, height=380, margin=dict(l=20, r=20, t=20, b=20),
        hoverlabel=dict(bgcolor="rgba(17,24,39,0.95)", font=dict(color="#F8FAFC")),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Detailed metrics table
    st.markdown('<div class="section-header">📋 Full Metrics Table</div>', unsafe_allow_html=True)
    display_cols = ["name", "accuracy", "precision", "recall", "f1", "auc_roc", "cv_f1_mean", "cv_f1_std"]
    df_display = df_m[display_cols].copy()
    df_display.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "CV F1 Mean", "CV F1 Std"]
    for col in df_display.columns[1:]:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    st.dataframe(df_display.sort_values("F1", ascending=False).reset_index(drop=True),
                 use_container_width=True, hide_index=True)

# ── Tab 2: Feature Engineering ─────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🔬 Engineered Feature Analysis</div>', unsafe_allow_html=True)

    features_info = [
        ("char_count", "Character Count", "Total characters per review"),
        ("word_count", "Word Count", "Total words per review"),
        ("avg_word_length", "Avg Word Length", "Average characters per word"),
        ("exclamation_count", "Exclamation Marks", "Number of '!' symbols"),
        ("capital_ratio", "Capital Letter Ratio", "Fraction of uppercase chars"),
    ]

    col1, col2 = st.columns(2)
    for i, (feat, title, desc) in enumerate(features_info):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            fig = feature_distribution(df.sample(min(3000, len(df)), random_state=i), feat, f"{title} by Class")
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Business Insights ───────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">💡 Key Findings</div>', unsafe_allow_html=True)

    insights = [
        ("🎯", "Perfect Balance", "Dataset has 50.0% Fake and 50.0% Real reviews — no class imbalance correction needed.", SUCCESS_COLOR),
        ("✍️", "Fake reviews are shorter", "Computer-generated reviews tend to have fewer characters and words on average.", DANGER_COLOR),
        ("❗", "Punctuation patterns", "Real reviews use more varied punctuation; fake ones are more uniform.", WARNING_COLOR),
        ("🔡", "Capital letters", "Human reviews show more expressive capitalization patterns.", PRIMARY_COLOR),
        ("⚡", "SVM wins", "Linear SVM achieves the best F1=0.897 with fast training (~2s vs 60s for RF).", SECONDARY_COLOR),
    ]

    for icon, title, body, color in insights:
        st.markdown(
            f"""
            <div class="glass-card" style="display:flex;gap:1rem;align-items:flex-start;margin-bottom:0.75rem;padding:1.25rem 1.5rem;">
                <div style="font-size:1.8rem;flex-shrink:0;">{icon}</div>
                <div>
                    <div style="font-weight:700;color:{color};margin-bottom:0.25rem;">{title}</div>
                    <div style="color:#94A3B8;font-size:0.9rem;">{body}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="app-footer">FakeGuard AI · Analytics</div>', unsafe_allow_html=True)
