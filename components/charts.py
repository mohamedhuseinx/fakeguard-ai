"""
components/charts.py
====================
Reusable Plotly chart builders for the FakeGuard AI dashboard.
All charts follow the dark-mode premium theme.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import (
    DANGER_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
)

# ── Base Layout ────────────────────────────────────────────────────────────────

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94A3B8", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(148,163,184,0.1)", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="rgba(148,163,184,0.1)", showgrid=True, zeroline=False),
    hoverlabel=dict(
        bgcolor="rgba(17,24,39,0.95)",
        bordercolor="rgba(108,99,255,0.5)",
        font=dict(family="Inter, sans-serif", color="#F8FAFC", size=13),
    ),
)

_COLOR_SEQUENCE = [PRIMARY_COLOR, SECONDARY_COLOR, WARNING_COLOR, DANGER_COLOR, "#A78BFA"]


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply dark-mode base layout to a Plotly figure."""
    fig.update_layout(**_BASE_LAYOUT, title=dict(
        text=title, font=dict(color="#F8FAFC", size=16, family="Inter, sans-serif"), x=0.02
    ))
    return fig


# ── Label Distribution ─────────────────────────────────────────────────────────

def class_distribution_pie(df: pd.DataFrame) -> go.Figure:
    """Donut chart of Fake vs Real review distribution."""
    counts = df["target"].value_counts()
    labels = ["Fake (CG)", "Real (OR)"]
    colors = [DANGER_COLOR, SUCCESS_COLOR]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts.values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="percent+label",
        textfont=dict(size=13, family="Inter, sans-serif", color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Review Class Distribution", font=dict(color="#F8FAFC", size=16), x=0.02),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1, font=dict(color="#94A3B8")),
        annotations=[dict(
            text=f"<b>{counts.sum():,}</b><br>Total",
            x=0.5, y=0.5, font=dict(size=16, color="#F8FAFC", family="Inter"),
            showarrow=False,
        )],
    )
    return fig


# ── Feature Distribution ───────────────────────────────────────────────────────

def feature_distribution(df: pd.DataFrame, feature: str, title: str) -> go.Figure:
    """Overlapping histogram of a numeric feature by class."""
    fig = go.Figure()
    styles = [(0, DANGER_COLOR, "Fake (CG)"), (1, SUCCESS_COLOR, "Real (OR)")]

    for cls, color, name in styles:
        subset = df[df["target"] == cls][feature].dropna()
        subset_clipped = subset[subset <= subset.quantile(0.99)]
        fig.add_trace(go.Histogram(
            x=subset_clipped,
            name=name,
            marker_color=color,
            opacity=0.7,
            nbinsx=50,
            hovertemplate=f"<b>{name}</b><br>{feature}: %{{x}}<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(**_BASE_LAYOUT,
        title=dict(text=title, font=dict(color="#F8FAFC", size=16), x=0.02),
        barmode="overlay",
        legend=dict(orientation="h", y=1.05, font=dict(color="#94A3B8")),
    )
    return fig


# ── Correlation Heatmap ────────────────────────────────────────────────────────

def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of numeric feature correlations."""
    numeric_cols = ["char_count", "word_count", "avg_word_length",
                    "exclamation_count", "question_count", "capital_ratio", "target"]
    corr = df[[c for c in numeric_cols if c in df.columns]].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[[0, DANGER_COLOR], [0.5, "rgba(17,24,39,1)"], [1, PRIMARY_COLOR]],
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    _apply_base(fig, "Feature Correlation Matrix")
    fig.update_layout(xaxis_tickangle=-35, height=420)
    return fig


# ── Model Comparison Bar ───────────────────────────────────────────────────────

def model_comparison_bar(metrics_list: List[Dict]) -> go.Figure:
    """Grouped bar chart comparing model metrics."""
    df = pd.DataFrame(metrics_list)
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    colors = [PRIMARY_COLOR, SECONDARY_COLOR, WARNING_COLOR, DANGER_COLOR]

    fig = go.Figure()
    for col, color in zip(metric_cols, colors):
        if col in df.columns:
            fig.add_trace(go.Bar(
                name=col.capitalize(),
                x=df["name"],
                y=df[col],
                marker_color=color,
                marker_line_width=0,
                opacity=0.9,
                hovertemplate=f"<b>%{{x}}</b><br>{col.capitalize()}: %{{y:.4f}}<extra></extra>",
            ))

    _apply_base(fig, "Model Performance Comparison")
    fig.update_layout(
        barmode="group",
        xaxis_tickangle=-20,
        yaxis_range=[0, 1.05],
        legend=dict(orientation="h", y=1.05, font=dict(color="#94A3B8")),
        height=420,
    )
    return fig


# ── Confusion Matrix ───────────────────────────────────────────────────────────

def confusion_matrix_heatmap(cm: List[List[int]], model_name: str = "") -> go.Figure:
    """Annotated confusion matrix heatmap."""
    labels = ["Fake (CG)", "Real (OR)"]
    cm_arr = np.array(cm)

    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=labels,
        y=labels,
        colorscale=[[0, "rgba(17,24,39,1)"], [0.5, "rgba(108,99,255,0.3)"], [1, PRIMARY_COLOR]],
        showscale=False,
        text=cm_arr,
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=20, color="white"),
        hovertemplate="<b>Actual: %{y}</b><br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    title = f"Confusion Matrix — {model_name}" if model_name else "Confusion Matrix"
    _apply_base(fig, title)
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=360,
    )
    return fig


# ── ROC Curve ─────────────────────────────────────────────────────────────────

def roc_curve_chart(fpr: np.ndarray, tpr: np.ndarray, auc: float, name: str = "") -> go.Figure:
    """ROC curve with AUC annotation."""
    fig = go.Figure()

    # Diagonal baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="rgba(148,163,184,0.4)", dash="dash", width=1),
        name="Random Baseline",
        showlegend=True,
    ))

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        line=dict(color=PRIMARY_COLOR, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.12)",
        name=f"{name} (AUC={auc:.4f})",
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
    ))

    _apply_base(fig, f"ROC Curve — AUC = {auc:.4f}")
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1.02], **_BASE_LAYOUT["xaxis"]),
        yaxis=dict(range=[0, 1.02], **_BASE_LAYOUT["yaxis"]),
        legend=dict(x=0.5, y=0.05, font=dict(color="#94A3B8")),
        height=400,
    )
    return fig


# ── Precision-Recall Curve ─────────────────────────────────────────────────────

def pr_curve_chart(precision: np.ndarray, recall: np.ndarray, name: str = "") -> go.Figure:
    """Precision-Recall curve."""
    fig = go.Figure(go.Scatter(
        x=recall, y=precision,
        mode="lines",
        line=dict(color=SECONDARY_COLOR, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.1)",
        name=name,
        hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
    ))
    _apply_base(fig, "Precision-Recall Curve")
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=400,
    )
    return fig


# ── Feature Importance ─────────────────────────────────────────────────────────

def feature_importance_bar(fi_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top feature importances by class."""
    if fi_df.empty:
        return go.Figure()

    fake_df = fi_df[fi_df["direction"] == "Fake"].head(top_n)
    real_df = fi_df[fi_df["direction"] == "Real"].head(top_n)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Top Fake-Leaning Words", "Top Real-Leaning Words"],
    )
    fig.add_trace(go.Bar(
        x=fake_df["importance"],
        y=fake_df["feature"],
        orientation="h",
        marker_color=DANGER_COLOR,
        marker_opacity=0.85,
        name="Fake",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=real_df["importance"],
        y=real_df["feature"],
        orientation="h",
        marker_color=SUCCESS_COLOR,
        marker_opacity=0.85,
        name="Real",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(**_BASE_LAYOUT, title=dict(
        text="Feature Importance (TF-IDF Coefficients)",
        font=dict(color="#F8FAFC", size=16), x=0.02,
    ), height=500, showlegend=False)
    return fig


# ── Confidence Gauge ───────────────────────────────────────────────────────────

def confidence_gauge(score: float, label: str) -> go.Figure:
    """Gauge chart for prediction confidence."""
    color = SUCCESS_COLOR if score >= 0.7 else (WARNING_COLOR if score >= 0.5 else DANGER_COLOR)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number=dict(suffix="%", font=dict(size=32, color="#F8FAFC", family="Inter")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#94A3B8", tickfont=dict(color="#94A3B8")),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(17,24,39,0.8)",
            bordercolor="rgba(148,163,184,0.15)",
            steps=[
                dict(range=[0, 50], color="rgba(255,71,87,0.08)"),
                dict(range=[50, 70], color="rgba(255,165,2,0.08)"),
                dict(range=[70, 100], color="rgba(46,213,115,0.08)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=score * 100),
        ),
        title=dict(text=f"Confidence — <b>{label}</b>", font=dict(color="#94A3B8", size=13, family="Inter")),
    ))
    fig.update_layout(**{**_BASE_LAYOUT, "margin": dict(l=20, r=20, t=20, b=20)}, height=260)
    return fig


# ── Word-Frequency Bar ─────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.3) -> str:
    """Convert a 6-digit hex color string to an rgba() string for Plotly colorscales."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def top_words_bar(word_freq: Dict[str, int], title: str, color: str = PRIMARY_COLOR) -> go.Figure:
    """Horizontal bar chart of top N words by frequency."""
    words = list(word_freq.keys())[:20]
    counts = [word_freq[w] for w in words]

    # Build valid rgba start color (Plotly 6 rejects 8-digit hex alpha notation)
    color_transparent = _hex_to_rgba(color, alpha=0.25)

    fig = go.Figure(go.Bar(
        x=counts[::-1],
        y=words[::-1],
        orientation="h",
        marker=dict(
            color=counts[::-1],
            colorscale=[[0, color_transparent], [1, color]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Frequency: %{x:,}<extra></extra>",
    ))
    _apply_base(fig, title)
    fig.update_layout(height=450)
    return fig
