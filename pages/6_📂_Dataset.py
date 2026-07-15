"""
pages/6_📂_Dataset.py — Searchable Dataset Browser
"""

import io

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dataset — FakeGuard AI", page_icon="📂", layout="wide")

from components.styles import inject_css
from config.settings import APP_VERSION
from utils.helpers import load_dataset

inject_css()

with st.sidebar:
    st.markdown("### 📂 Dataset Browser")
    st.markdown(f"*v{APP_VERSION}*")
    st.divider()
    st.markdown("**Filters**")
    filter_class = st.multiselect("Class", ["Fake (CG)", "Real (OR)"], default=["Fake (CG)", "Real (OR)"])
    min_words = st.slider("Min word count", 1, 200, 1)
    max_words = st.slider("Max word count", 10, 1000, 500)
    search_query = st.text_input("🔍 Search text", placeholder="keyword...")

st.markdown(
    """
    <div class="hero-section" style="padding:2rem;">
        <h1 style="font-size:2rem;font-weight:800;color:#F8FAFC;margin:0 0 0.5rem;">
            📂 Dataset Browser
        </h1>
        <p style="color:#94A3B8;margin:0;">
            Browse, filter, and download the fake reviews dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

with st.spinner("Loading dataset..."):
    df = load_dataset()

# ── Apply Filters ──────────────────────────────────────────────────────────────
filtered = df.copy()

if filter_class:
    filtered = filtered[filtered["label_name"].isin(filter_class)]

filtered = filtered[
    (filtered["word_count"] >= min_words) &
    (filtered["word_count"] <= max_words)
]

if search_query.strip():
    mask = filtered["text"].str.contains(search_query, case=False, na=False)
    filtered = filtered[mask]

# ── Summary ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Showing", f"{len(filtered):,}", f"of {len(df):,} total")
c2.metric("🔴 Fake", f"{(filtered.target==0).sum():,}")
c3.metric("🟢 Real", f"{(filtered.target==1).sum():,}")
c4.metric("📏 Avg Words", f"{filtered['word_count'].mean():.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Table ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Review Records</div>', unsafe_allow_html=True)

display_cols = ["label_name", "text", "word_count", "char_count", "exclamation_count"]
show_df = filtered[display_cols].rename(columns={
    "label_name": "Class",
    "text": "Review Text",
    "word_count": "Words",
    "char_count": "Chars",
    "exclamation_count": "Exclamations",
}).reset_index(drop=True)

# Truncate long text for display
show_df["Review Text"] = show_df["Review Text"].str[:150] + "..."

page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=1)
total_pages = max(1, (len(show_df) - 1) // page_size + 1)
page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
start = (page_num - 1) * page_size
end = min(start + page_size, len(show_df))

st.caption(f"Showing rows {start+1}–{end} of {len(show_df):,}")
st.dataframe(show_df.iloc[start:end], use_container_width=True, hide_index=True, height=420)

# ── Download ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_dl1, col_dl2, _ = st.columns([1.2, 1.2, 4])

with col_dl1:
    csv_buf = io.StringIO()
    filtered[["label_name", "text", "word_count", "char_count"]].to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download Filtered CSV",
        csv_buf.getvalue(),
        "fakeguard_filtered.csv",
        "text/csv",
        use_container_width=True,
    )

with col_dl2:
    csv_buf2 = io.StringIO()
    df[["label_name", "text", "word_count", "char_count"]].to_csv(csv_buf2, index=False)
    st.download_button(
        "⬇️ Download Full Dataset",
        csv_buf2.getvalue(),
        "fakeguard_full.csv",
        "text/csv",
        use_container_width=True,
    )

# ── Sample Viewer ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🔎 Full Review Viewer — click to expand a single review"):
    if len(filtered) > 0:
        idx = st.number_input("Review index (0-based)", min_value=0, max_value=len(filtered)-1, value=0)
        row = filtered.iloc[idx]
        cls_color = "#FF4757" if row["target"] == 0 else "#2ED573"
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display:flex;gap:0.75rem;margin-bottom:1rem;flex-wrap:wrap;">
                    <span style="color:{cls_color};font-weight:700;font-size:1.1rem;">
                        {'🔴 Fake (CG)' if row['target']==0 else '🟢 Real (OR)'}
                    </span>
                    <span class="stat-badge">📏 {row['word_count']} words</span>
                    <span class="stat-badge">🔡 {row['char_count']} chars</span>
                </div>
                <div style="color:#F8FAFC;line-height:1.8;font-size:0.95rem;">
                    {row['text']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No reviews match the current filters.")

st.markdown('<div class="app-footer">FakeGuard AI · Dataset Browser</div>', unsafe_allow_html=True)
