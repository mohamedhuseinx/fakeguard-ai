#!/bin/bash
# setup.sh — Bootstrap script for Streamlit Cloud / fresh environments

set -e

echo "=== FakeGuard AI Setup ==="

echo "1. Installing Python dependencies..."
pip install -r requirements.txt

echo "2. Downloading NLTK resources..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
print('NLTK resources ready.')
"

echo "3. Training model pipeline (if not exists)..."
if [ ! -f "models/saved/pipeline.joblib" ]; then
    python models/trainer.py
else
    echo "Pipeline already exists, skipping training."
fi

echo ""
echo "=== Setup Complete! ==="
echo "Run: streamlit run app.py"
