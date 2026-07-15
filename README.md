# 🛡️ FakeGuard AI — Fake Review Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-6.3-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](.github/workflows/ci.yml)

**Advanced NLP-powered fake review detection with a premium SaaS-grade dashboard.**

[🚀 Live Demo](#) · [📖 Documentation](#architecture) · [🐛 Report Bug](../../issues)

</div>

---

## 📌 Project Overview

FakeGuard AI is a production-grade NLP application that detects computer-generated (fake) reviews from genuine human-written ones. It achieves **F1 = 0.897** and **AUC = 0.963** using a Linear SVM on TF-IDF features.

Built as a portfolio-quality project demonstrating end-to-end ML engineering:
- Data preprocessing and feature engineering
- Multi-model comparison with full evaluation metrics
- Production Streamlit dashboard with 8 interactive pages
- Modular, PEP8-compliant, type-annotated Python codebase
- Docker + CI/CD deployment ready

---

## 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|----|---------|
| 🏆 **SVM** | 0.8972 | 0.8973 | 0.8975 | **0.8974** | **0.9629** |
| 🥈 Logistic Regression | 0.8936 | 0.8933 | 0.8937 | 0.8935 | 0.9617 |
| 🥉 Naive Bayes | 0.8653 | 0.8647 | 0.8679 | 0.8663 | 0.9458 |
| Random Forest | 0.8085 | 0.8098 | 0.8274 | 0.8185 | 0.9019 |
| Decision Tree | 0.6740 | 0.6905 | 0.7358 | 0.7124 | 0.7352 |

*Evaluated on 20% held-out test set (8,083 samples) with 5-fold stratified cross-validation.*

---

## ✨ Features

- **🤖 Multi-Model Ensemble** — NB, LR, DT, RF, SVM with majority voting
- **📊 Full EDA Dashboard** — Class distribution, word clouds, feature correlations
- **🔍 Explainability** — TF-IDF coefficient analysis and SHAP values
- **📁 Batch Prediction** — CSV upload with downloadable results
- **📈 Interactive Charts** — Plotly confusion matrix, ROC/PR curves, feature importance
- **🎨 Premium UI** — Glassmorphism dark-mode Streamlit interface
- **⚡ Optimized Pipeline** — O(1) stopword lookup, cached stemming (10x faster)
- **🐳 Docker Ready** — Single command deployment
- **🔄 CI/CD** — GitHub Actions with multi-Python testing

---

## 🏗️ Architecture

```
NLP Project/
├── app.py                     # Main Streamlit entry point
├── config/settings.py         # Centralized configuration
├── data/                      # Dataset storage
├── models/
│   ├── trainer.py             # Training pipeline (cross-val + grid search)
│   ├── predictor.py           # Inference engine (single + batch + SHAP)
│   └── saved/pipeline.joblib  # Trained model (TF-IDF + SVM)
├── utils/
│   ├── text_processor.py      # Optimized NLP preprocessing
│   └── helpers.py             # Data loading + caching utilities
├── components/
│   ├── charts.py              # Plotly chart builders
│   └── styles.py              # Premium CSS injection
├── pages/                     # 8 Streamlit dashboard pages
├── tests/                     # Unit tests (pytest)
├── Dockerfile
└── .github/workflows/ci.yml   # CI/CD pipeline
```

---

## 🚀 Quick Start

### Option 1: Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/MOHAMED/fakeguard-ai.git
cd fakeguard-ai

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# 5. Add dataset (get from Kaggle: "Fake Reviews Dataset")
# Place in: data/fake reviews dataset.csv

# 6. Train the model
python models/trainer.py

# 7. Launch the app
streamlit run app.py
```

### Option 2: Docker

```bash
# Build and run
docker-compose up --build

# App available at: http://localhost:8501
```

---

## 📱 Application Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | KPI dashboard, model leaderboard, dataset overview |
| 📊 **Data Exploration** | EDA: distributions, word frequency, correlation heatmap |
| 📈 **Analytics** | Model comparison charts, feature engineering, business insights |
| 🤖 **Prediction** | Single review + CSV batch prediction with confidence scores |
| 📉 **Model Performance** | Confusion matrix, ROC curve, classification report |
| 📂 **Dataset** | Searchable, filterable dataset browser with download |
| ⚙️ **Settings** | Configuration, theme, system info |
| 📘 **About** | Project overview, architecture, author |

---

## 🔬 ML Pipeline

```
Raw CSV (40,412 reviews)
    ↓ Load & Clean (deduplicate, encode labels CG→0, OR→1)
    ↓ Feature Engineering (char_count, word_count, capital_ratio, ...)
    ↓ Text Preprocessing
      └─ Lowercase → Tokenize → Remove stopwords (frozenset O(1))
         → Filter alphanumeric → Porter stem (lru_cached)
    ↓ TF-IDF (5000 features, unigrams+bigrams, sublinear_tf=True)
    ↓ Train 5 Classifiers + 5-Fold StratifiedKFold CV
    ↓ Evaluate (Accuracy, F1, Precision, Recall, AUC-ROC, Confusion Matrix)
    ↓ Save Best Pipeline → models/saved/pipeline.joblib (joblib compress=3)
```

---

## 🔧 Key Technical Improvements

### Performance Optimization
- **O(n²) → O(1):** Stopwords stored in `frozenset` instead of list
- **Cached stemming:** `@lru_cache(maxsize=10_000)` on `PorterStemmer.stem()`
- **Sublinear TF-IDF:** `log(1+tf)` scaling improves text classification
- **Bigrams:** Added `ngram_range=(1,2)` to capture phrase patterns

### Code Quality
- Full type hints throughout all modules
- Docstrings on every public function
- PEP8 compliant (max line 100)
- SOLID principles applied to class design
- DRY — no duplicate preprocessing code

### ML Best Practices
- `StratifiedKFold` (5 folds) instead of single train/test split
- Balanced dataset — no class imbalance correction needed
- Unified `sklearn.Pipeline` saves vectorizer+model together
- `joblib.dump(compress=3)` for efficient serialization

---

## 📊 Dataset

- **Source:** [Fake Reviews Dataset](https://kaggle.com/datasets) (Kaggle)
- **Size:** 40,412 reviews
- **Balance:** 50.0% Fake (CG), 50.0% Real (OR)
- **Features:** Raw review text, label (CG/OR)

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.13 | Core language |
| Scikit-learn | 1.7 | ML models and pipeline |
| NLTK | 3.9 | Text preprocessing |
| Streamlit | 1.51 | Web application framework |
| Plotly | 6.3 | Interactive visualizations |
| Pandas | 2.3 | Data manipulation |
| NumPy | 2.3 | Numerical operations |
| Joblib | 1.5 | Model serialization |
| SHAP | 0.46 | Model explainability |

---

## 🧪 Running Tests

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=utils --cov=models --cov-report=html
```

---

## 🔮 Future Work

- [ ] Fine-tuned BERT / DistilBERT for higher accuracy (~95%+)
- [ ] REST API endpoint (FastAPI) for programmatic access
- [ ] Real-time review scraping integration
- [ ] User authentication and review history
- [ ] Multi-language support
- [ ] Active learning with user feedback loop

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Dataset: Fake Reviews Dataset from Kaggle
- Libraries: scikit-learn, NLTK, Streamlit, Plotly teams
- Design inspiration: Modern SaaS dashboards and glassmorphism UI trends

---

<div align="center">

**Built with ❤️ by Mohamed Hussein**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mohamedhuseinx)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohamedhusseinx/)

⭐ **Star this repo if you found it useful!** ⭐

</div>
