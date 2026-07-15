"""
pages/8_📘_About.py — Project & Author Information
"""

import streamlit as st

st.set_page_config(page_title="About — FakeGuard AI", page_icon="📘", layout="wide")

from components.styles import inject_css
from config.settings import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    AUTHOR_EMAIL,
    AUTHOR_GITHUB,
    AUTHOR_LINKEDIN,
    AUTHOR_NAME,
)

inject_css()

with st.sidebar:
    st.markdown("### 📘 About")
    st.markdown(f"*v{APP_VERSION}*")

st.markdown(
    f"""
    <div class="hero-section">
        <div style="font-size:3rem;margin-bottom:1rem;">🛡️</div>
        <h1 style="font-size:2.5rem;font-weight:900;margin:0 0 0.5rem;
                   background:linear-gradient(135deg,#6C63FF,#00D4AA);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            {APP_NAME}
        </h1>
        <p style="color:#94A3B8;font-size:1.1rem;max-width:700px;margin:0 auto;">
            {APP_TAGLINE}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📖 Project Overview", "🏗️ Architecture", "👤 Author"])

# ── Tab 1: Project Overview ────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            <div class="section-header">📖 What is FakeGuard AI?</div>
            <div class="glass-card">
                <p style="color:#94A3B8;line-height:1.8;font-size:0.95rem;">
                    <strong style="color:#F8FAFC;">FakeGuard AI</strong> is an advanced NLP-powered system that detects
                    computer-generated (fake) reviews from genuine ones. It was built using a dataset of 40,000+ labeled
                    reviews (CG = Computer Generated, OR = Original).
                </p>
                <p style="color:#94A3B8;line-height:1.8;font-size:0.95rem;">
                    The system uses <strong style="color:#6C63FF;">TF-IDF vectorization</strong> with unigram + bigram
                    features and evaluates 5 machine learning classifiers. The best model (Linear SVM) achieves
                    <strong style="color:#2ED573;">F1 = 0.897</strong> and <strong style="color:#2ED573;">AUC = 0.963</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">✨ Features</div>', unsafe_allow_html=True)

        features = [
            ("🤖", "5 ML Models", "NB, LR, DT, RF, SVM compared with full metrics"),
            ("📊", "Full EDA", "Word clouds, distributions, correlation analysis"),
            ("🔍", "Explainability", "TF-IDF coefficient analysis & SHAP values"),
            ("📁", "Batch Prediction", "CSV upload for bulk review classification"),
            ("📈", "Live Dashboard", "Interactive Plotly charts, confusion matrix, ROC"),
            ("🎨", "Premium UI", "Glassmorphism dark-mode Streamlit interface"),
        ]
        for icon, title, desc in features:
            st.markdown(
                f"""
                <div style="display:flex;gap:0.75rem;align-items:flex-start;padding:0.6rem 0;
                            border-bottom:1px solid rgba(148,163,184,0.08);">
                    <span style="font-size:1.2rem;">{icon}</span>
                    <div>
                        <span style="color:#F8FAFC;font-weight:600;">{title}</span>
                        <span style="color:#64748B;font-size:0.85rem;"> — {desc}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown('<div class="section-header">📊 Model Results</div>', unsafe_allow_html=True)
        results = [
            ("SVM", "89.7%", "96.3%", "🏆"),
            ("LR", "89.4%", "96.2%", "🥈"),
            ("NB", "86.5%", "94.6%", "🥉"),
            ("RF", "80.9%", "90.2%", "4️⃣"),
            ("DT", "67.4%", "73.5%", "5️⃣"),
        ]
        for model, f1, auc, medal in results:
            st.markdown(
                f"""
                <div style="background:rgba(17,24,39,0.7);border:1px solid rgba(148,163,184,0.1);
                            border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.5rem;
                            display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:1rem;">{medal}</span>
                    <span style="color:#F8FAFC;font-weight:600;">{model}</span>
                    <span style="color:#6C63FF;font-size:0.85rem;">F1 {f1}</span>
                    <span style="color:#00D4AA;font-size:0.85rem;">AUC {auc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🛠️ Tech Stack</div>', unsafe_allow_html=True)
        tech_stack = [
            ("🐍", "Python 3.13"),
            ("🤖", "Scikit-learn 1.7"),
            ("📊", "Streamlit 1.51"),
            ("📈", "Plotly 6.3"),
            ("🐼", "Pandas 2.3"),
            ("🔢", "NumPy 2.3"),
            ("📝", "NLTK 3.9"),
            ("💾", "Joblib 1.5"),
        ]
        for icon, tech in tech_stack:
            st.markdown(
                f'<span class="stat-badge">{icon} {tech}</span>',
                unsafe_allow_html=True,
            )

# ── Tab 2: Architecture ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🏗️ Project Architecture</div>', unsafe_allow_html=True)
    st.code(
        """
NLP Project/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Professional documentation
├── Dockerfile                      # Container deployment
├── .streamlit/config.toml          # Streamlit theme config
│
├── config/
│   └── settings.py                 # All constants & configuration
│
├── data/
│   └── fake reviews dataset.csv   # Raw dataset (40,412 reviews)
│
├── models/
│   ├── trainer.py                  # ML training pipeline
│   ├── predictor.py                # Inference engine
│   └── saved/
│       ├── pipeline.joblib         # Trained model pipeline
│       └── model_metrics.json     # Evaluation results
│
├── utils/
│   ├── text_processor.py           # NLP preprocessing (optimized)
│   └── helpers.py                  # Data loading, persistence
│
├── components/
│   ├── charts.py                   # Plotly chart builders
│   └── styles.py                   # Premium CSS injection
│
├── pages/                          # Streamlit multi-page app
│   ├── 1_🏠_Home.py
│   ├── 2_📊_Data_Exploration.py
│   ├── 3_📈_Analytics.py
│   ├── 4_🤖_Prediction.py
│   ├── 5_📉_Model_Performance.py
│   ├── 6_📂_Dataset.py
│   ├── 7_⚙️_Settings.py
│   └── 8_📘_About.py
│
└── tests/
    ├── test_predictor.py
    └── test_text_processor.py
        """,
        language="text",
    )

    st.markdown('<div class="section-header">🔄 ML Pipeline Flow</div>', unsafe_allow_html=True)
    st.code(
        """
Raw CSV Data
    ↓
Load & Clean (drop duplicates, encode labels)
    ↓
Feature Engineering (char_count, word_count, capital_ratio, ...)
    ↓
Text Preprocessing:
  lowercase → tokenize → remove stopwords (O(1) set lookup)
  → filter alphanumeric → Porter stem
    ↓
TF-IDF Vectorization (5000 features, unigrams+bigrams, sublinear_tf)
    ↓
Train 5 Classifiers (NB, LR, DT, RF, SVM)
  + 5-Fold Stratified Cross-Validation
    ↓
Evaluate (Accuracy, F1, Precision, Recall, AUC-ROC, Confusion Matrix)
    ↓
Save Best Pipeline → pipeline.joblib (compressed, joblib)
Save Metrics → model_metrics.json
        """,
        language="text",
    )

# ── Tab 3: Author ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        f"""
        <div class="glass-card" style="max-width:600px;margin:0 auto;text-align:center;padding:2.5rem;">
            <div style="width:80px;height:80px;border-radius:50%;
                        background:linear-gradient(135deg,#6C63FF,#00D4AA);
                        display:flex;align-items:center;justify-content:center;
                        margin:0 auto 1.25rem;font-size:2rem;">👨‍💻</div>
            <div style="font-size:1.6rem;font-weight:800;color:#F8FAFC;margin-bottom:0.25rem;">{AUTHOR_NAME}</div>
            <div style="color:#6C63FF;font-size:0.9rem;font-weight:500;margin-bottom:1.5rem;letter-spacing:0.05em;">
                AI Engineer · Data Scientist · NLP Specialist
            </div>
            <div style="display:flex;gap:0.75rem;justify-content:center;margin-bottom:1.5rem;">
                <a href="{AUTHOR_GITHUB}" target="_blank"
                   style="background:rgba(108,99,255,0.15);border:1px solid rgba(108,99,255,0.3);
                          color:#6C63FF;border-radius:8px;padding:0.5rem 1.25rem;
                          text-decoration:none;font-weight:600;font-size:0.9rem;transition:all 0.3s;">
                    ⭐ GitHub
                </a>
                <a href="{AUTHOR_LINKEDIN}" target="_blank"
                   style="background:rgba(0,212,170,0.15);border:1px solid rgba(0,212,170,0.3);
                          color:#00D4AA;border-radius:8px;padding:0.5rem 1.25rem;
                          text-decoration:none;font-weight:600;font-size:0.9rem;transition:all 0.3s;">
                    💼 LinkedIn
                </a>
                <a href="mailto:{AUTHOR_EMAIL}"
                   style="background:rgba(255,165,2,0.15);border:1px solid rgba(255,165,2,0.3);
                          color:#FFA502;border-radius:8px;padding:0.5rem 1.25rem;
                          text-decoration:none;font-weight:600;font-size:0.9rem;transition:all 0.3s;">
                    📧 Email
                </a>
            </div>
            <div style="color:#64748B;font-size:0.85rem;line-height:1.7;">
                Built {APP_NAME} as a production-grade NLP application demonstrating<br>
                end-to-end ML engineering — from raw data to deployed AI SaaS.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📄 License & Acknowledgements</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
            <div style="color:#94A3B8;font-size:0.9rem;line-height:1.8;">
                <p>📜 <strong style="color:#F8FAFC;">License:</strong> MIT — free to use, modify, and distribute.</p>
                <p>🙏 <strong style="color:#F8FAFC;">Dataset:</strong> Fake Reviews Dataset from Kaggle/UCI ML Repository.</p>
                <p>🛠️ <strong style="color:#F8FAFC;">Libraries:</strong> scikit-learn, NLTK, Streamlit, Plotly, Pandas, NumPy, Joblib.</p>
                <p>🎨 <strong style="color:#F8FAFC;">Design:</strong> Glassmorphism dark-mode inspired by modern SaaS dashboards.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="app-footer">FakeGuard AI · About</div>', unsafe_allow_html=True)
