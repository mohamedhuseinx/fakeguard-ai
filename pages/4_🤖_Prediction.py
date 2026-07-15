"""
pages/4_🤖_Prediction.py — AI Prediction Interface
"""

import io
import time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediction — FakeGuard AI", page_icon="🤖", layout="wide")

from components.charts import confidence_gauge, feature_importance_bar
from components.styles import inject_css
from config.settings import APP_VERSION, DANGER_COLOR, PRIMARY_COLOR, SUCCESS_COLOR, WARNING_COLOR
from models.predictor import ReviewPredictor
from utils.helpers import confidence_color

inject_css()

# ── Load predictor (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_predictor() -> ReviewPredictor:
    return ReviewPredictor()

with st.sidebar:
    st.markdown("### 🤖 Prediction")
    st.markdown(f"*v{APP_VERSION}*")
    st.divider()
    st.markdown("**Prediction Mode**")
    mode = st.radio("", ["Single Review", "Batch (CSV Upload)"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Example reviews:**")
    examples = {
        "Typical fake": "This product is absolutely amazing! Best purchase ever! 5 stars!!! Love it so much!!",
        "Typical real": "I've been using this for about 3 weeks. The build quality is decent but the battery life is shorter than advertised. Good for the price.",
        "Ambiguous": "Works as expected. Fast delivery. Would buy again.",
    }
    chosen_example = st.selectbox("Load example", ["(none)"] + list(examples.keys()))

st.markdown(
    """
    <div class="hero-section" style="padding:2rem;">
        <h1 style="font-size:2rem;font-weight:800;color:#F8FAFC;margin:0 0 0.5rem;">
            🤖 AI Prediction Engine
        </h1>
        <p style="color:#94A3B8;margin:0;">
            Paste a review below to instantly detect whether it's fake or genuine.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Load predictor ─────────────────────────────────────────────────────────────
try:
    predictor = get_predictor()
except FileNotFoundError:
    st.error("⚠️ Model not found. Run `python models/trainer.py` first.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE PREDICTION MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single Review":
    default_text = examples.get(chosen_example, "") if chosen_example != "(none)" else ""

    col_input, col_tips = st.columns([3, 1])
    with col_input:
        st.markdown('<div class="section-header">📝 Enter Review Text</div>', unsafe_allow_html=True)
        review_text = st.text_area(
            "Review",
            value=default_text,
            height=160,
            placeholder="Paste or type a review here (e.g., product review, restaurant review, hotel review)...",
            label_visibility="collapsed",
        )
        char_count = len(review_text)
        word_count = len(review_text.split()) if review_text.strip() else 0
        st.markdown(
            f'<div style="color:#64748B;font-size:0.8rem;margin-top:0.25rem;">'
            f'{char_count} chars · {word_count} words</div>',
            unsafe_allow_html=True,
        )

    with col_tips:
        st.markdown('<div class="section-header">💡 Tips</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="glass-card" style="padding:1rem;">
                <div style="color:#64748B;font-size:0.82rem;line-height:1.7;">
                    🔴 <b>Fake signals:</b><br>
                    · Excessive exclamation marks<br>
                    · Generic praise only<br>
                    · Very short/very long<br>
                    · Repeated superlatives<br><br>
                    🟢 <b>Real signals:</b><br>
                    · Specific details<br>
                    · Pros AND cons<br>
                    · Temporal references<br>
                    · Natural language
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        analyze_btn = st.button("🔍 Analyze Review", type="primary", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    if analyze_btn:
        if not review_text.strip():
            st.warning("Please enter a review to analyze.")
        elif len(review_text.strip()) < 10:
            st.warning("Review is too short. Please enter at least 10 characters.")
        else:
            with st.spinner("Analyzing review..."):
                time.sleep(0.3)  # subtle UX pause
                try:
                    result = predictor.predict_single(review_text)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            is_fake = result["label_int"] == 0
            label = result["label"]
            confidence = result["confidence"]
            probs = result.get("probabilities", {})
            votes = result.get("all_model_votes", {})

            # ── Result Banner ────────────────────────────────────────────────
            badge_class = "badge-fake" if is_fake else "badge-real"
            badge_text = "🔴 FAKE REVIEW DETECTED" if is_fake else "🟢 GENUINE REVIEW DETECTED"
            border_color = DANGER_COLOR if is_fake else SUCCESS_COLOR
            glow = "rgba(255,71,87,0.2)" if is_fake else "rgba(46,213,115,0.2)"

            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,{border_color}15,{border_color}08);
                            border:2px solid {border_color}60;border-radius:20px;
                            padding:2rem;text-align:center;margin:1.5rem 0;
                            box-shadow:0 8px 32px {glow};animation:fadeInUp 0.5s ease;">
                    <div class="{badge_class}" style="font-size:1.3rem;padding:0.6rem 2rem;margin-bottom:1rem;">
                        {badge_text}
                    </div>
                    <div style="color:#94A3B8;font-size:0.9rem;margin-top:0.5rem;">
                        Confidence: <strong style="color:{confidence_color(confidence)};">{confidence:.1%}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Metrics Row ──────────────────────────────────────────────────
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("🎯 Prediction", label)
            with r2:
                st.metric("📊 Confidence", f"{confidence:.1%}")
            with r3:
                fake_p = probs.get("Fake", 0)
                real_p = probs.get("Real", 0)
                st.metric("⚖️ Fake / Real Prob", f"{fake_p:.1%} / {real_p:.1%}")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Confidence Gauge + Model Votes ───────────────────────────────
            col_gauge, col_votes = st.columns([1, 1])

            with col_gauge:
                st.markdown('<div class="section-header">📊 Confidence Score</div>', unsafe_allow_html=True)
                st.plotly_chart(confidence_gauge(confidence, label), use_container_width=True)

                # Prob bars
                if probs:
                    st.markdown('<div class="section-header">🎲 Class Probabilities</div>', unsafe_allow_html=True)
                    for cls_name, prob in probs.items():
                        color = DANGER_COLOR if cls_name == "Fake" else SUCCESS_COLOR
                        st.markdown(
                            f"""
                            <div style="margin-bottom:0.6rem;">
                                <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                                    <span style="color:#94A3B8;font-size:0.85rem;">{cls_name}</span>
                                    <strong style="color:{color};">{prob:.1%}</strong>
                                </div>
                                <div style="background:rgba(148,163,184,0.1);border-radius:100px;height:8px;">
                                    <div style="background:{color};width:{prob*100:.1f}%;height:100%;border-radius:100px;
                                                transition:width 1s ease;"></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with col_votes:
                st.markdown('<div class="section-header">🗳️ Model Votes</div>', unsafe_allow_html=True)
                if votes:
                    fake_votes = sum(1 for v in votes.values() if v == "Fake")
                    real_votes = sum(1 for v in votes.values() if v == "Real")
                    majority = "Fake" if fake_votes > real_votes else "Real"
                    majority_color = DANGER_COLOR if majority == "Fake" else SUCCESS_COLOR

                    st.markdown(
                        f"""
                        <div style="background:rgba(17,24,39,0.7);border:1px solid rgba(148,163,184,0.15);
                                    border-radius:12px;padding:0.75rem 1rem;margin-bottom:0.75rem;
                                    display:flex;justify-content:space-between;align-items:center;">
                            <span style="color:#94A3B8;font-size:0.85rem;">Majority Vote</span>
                            <strong style="color:{majority_color};">{majority} ({fake_votes}F / {real_votes}R)</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    for model_name, vote in votes.items():
                        is_v_fake = vote == "Fake"
                        v_color = DANGER_COLOR if is_v_fake else SUCCESS_COLOR
                        v_icon = "🔴" if is_v_fake else "🟢"
                        border_cls = "vote-fake" if is_v_fake else "vote-real"
                        st.markdown(
                            f"""
                            <div class="vote-card {border_cls}">
                                <span style="color:#94A3B8;font-size:0.85rem;">{model_name}</span>
                                <span style="color:{v_color};font-weight:700;font-size:0.85rem;">{v_icon} {vote}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # ── SHAP Explanation ─────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🔍 Explainability — Feature Impact (SHAP / Coefficients)", expanded=False):
                with st.spinner("Computing explanation..."):
                    fi_df = predictor.get_feature_importance(top_n=20)
                if not fi_df.empty:
                    st.plotly_chart(feature_importance_bar(fi_df), use_container_width=True)
                    st.caption(
                        "Positive coefficients → words pushing toward Real. "
                        "Negative → pushing toward Fake."
                    )
                else:
                    st.info("Feature importance not available for this model type.")

            # ── Preprocessed Text ────────────────────────────────────────────
            with st.expander("⚙️ Preprocessed Text", expanded=False):
                st.code(result.get("clean_text", ""), language=None)
                st.caption("Text after lowercasing, tokenization, stopword removal, and Porter stemming.")

# ══════════════════════════════════════════════════════════════════════════════
# BATCH PREDICTION MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-header">📁 Batch Prediction — CSV Upload</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card" style="padding:1.25rem 1.5rem;margin-bottom:1rem;">
            <div style="color:#94A3B8;font-size:0.9rem;">
                Upload a CSV file with a column named <code style="color:#6C63FF;">text</code> containing review texts.<br>
                The app will predict each row and return results as a downloadable CSV.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Must contain a 'text' column",
        label_visibility="collapsed",
    )

    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        if "text" not in batch_df.columns:
            st.error("CSV must have a column named `text`.")
            st.stop()

        st.info(f"Loaded **{len(batch_df):,}** reviews. Running predictions...")
        progress = st.progress(0, text="Analyzing...")

        with st.spinner("Processing batch..."):
            texts = batch_df["text"].fillna("").tolist()
            results_df = predictor.predict_batch(texts)
            progress.progress(100, text="Done ✅")

        # ── Results Table ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

        fake_count = (results_df["label_int"] == 0).sum()
        real_count = (results_df["label_int"] == 1).sum()

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total Analyzed", f"{len(results_df):,}")
        mc2.metric("🔴 Fake Detected", f"{fake_count:,}", f"{fake_count/len(results_df):.1%}")
        mc3.metric("🟢 Real Detected", f"{real_count:,}", f"{real_count/len(results_df):.1%}")

        display_df = results_df[["text", "prediction", "confidence"]].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download button
        csv_buffer = io.StringIO()
        results_df.drop(columns=["label_int", "clean_text", "confidence_raw"], errors="ignore").to_csv(
            csv_buffer, index=False
        )
        st.download_button(
            "⬇️ Download Results as CSV",
            csv_buffer.getvalue(),
            "fakeguard_predictions.csv",
            "text/csv",
            use_container_width=False,
        )

st.markdown('<div class="app-footer">FakeGuard AI · Prediction Engine</div>', unsafe_allow_html=True)
