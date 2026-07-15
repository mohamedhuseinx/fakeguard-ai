"""
components/styles.py
====================
Premium glassmorphism dark-mode CSS for FakeGuard AI.
Injected via st.markdown() in every page.
"""

from config.settings import (
    BG_CARD,
    BG_DARK,
    BG_GLASS,
    DANGER_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
)

GLOBAL_CSS = f"""
<style>
/* ── Google Fonts ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ──────────────────────────────────────────────── */
:root {{
  --primary:    {PRIMARY_COLOR};
  --secondary:  {SECONDARY_COLOR};
  --danger:     {DANGER_COLOR};
  --success:    {SUCCESS_COLOR};
  --warning:    {WARNING_COLOR};
  --bg-dark:    {BG_DARK};
  --bg-card:    {BG_CARD};
  --bg-glass:   {BG_GLASS};
  --text-primary:   #F8FAFC;
  --text-secondary: #94A3B8;
  --border-subtle:  rgba(148,163,184,0.12);
  --glow-primary:   rgba(108,99,255,0.25);
  --glow-secondary: rgba(0,212,170,0.25);
  --radius-sm:  8px;
  --radius-md:  16px;
  --radius-lg:  24px;
  --transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}}

/* ── Reset & Base ────────────────────────────────────────────────── */
* {{ box-sizing: border-box; }}

.stApp {{
  background: linear-gradient(135deg, {BG_DARK} 0%, #0D1020 50%, #0A0E1A 100%) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--text-primary) !important;
}}

/* ── Hide default Streamlit chrome ─────────────────────────────── */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0D1020 0%, #111827 100%) !important;
  border-right: 1px solid var(--border-subtle) !important;
}}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {{
  color: var(--text-primary) !important;
}}
[data-testid="stSidebarNav"] {{
  padding-top: 0.5rem;
}}
[data-testid="stSidebarNav"] a {{
  border-radius: var(--radius-sm) !important;
  margin: 2px 8px !important;
  transition: var(--transition) !important;
}}
[data-testid="stSidebarNav"] a:hover {{
  background: rgba(108,99,255,0.15) !important;
  transform: translateX(4px);
}}

/* ── Metric Cards ────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
  background: linear-gradient(135deg, rgba(17,24,39,0.9) 0%, rgba(17,24,39,0.7) 100%);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md) !important;
  padding: 1.2rem 1.5rem !important;
  backdrop-filter: blur(12px);
  transition: var(--transition);
  box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}}
[data-testid="stMetric"]:hover {{
  border-color: rgba(108,99,255,0.4);
  box-shadow: 0 8px 32px var(--glow-primary);
  transform: translateY(-2px);
}}
[data-testid="stMetricLabel"] {{ color: var(--text-secondary) !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.08em; }}
[data-testid="stMetricValue"] {{ color: var(--text-primary) !important; font-weight: 700 !important; font-size: 1.8rem !important; }}
[data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; }}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button {{
  background: linear-gradient(135deg, var(--primary), #8B7CF8) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  padding: 0.6rem 1.4rem !important;
  transition: var(--transition) !important;
  box-shadow: 0 4px 15px var(--glow-primary) !important;
  letter-spacing: 0.02em;
}}
.stButton > button:hover {{
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px var(--glow-primary) !important;
  filter: brightness(1.1);
}}
.stButton > button:active {{ transform: translateY(0px) !important; }}

/* ── Text Inputs & Textareas ─────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {{
  background: rgba(17,24,39,0.8) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
  transition: var(--transition) !important;
}}
.stTextArea textarea:focus, .stTextInput input:focus {{
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px var(--glow-primary) !important;
}}

/* ── Select boxes ────────────────────────────────────────────────── */
.stSelectbox select, [data-baseweb="select"] {{
  background: rgba(17,24,39,0.8) !important;
  border-color: var(--border-subtle) !important;
  color: var(--text-primary) !important;
  border-radius: var(--radius-sm) !important;
}}

/* ── Sliders ─────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] {{
  color: var(--primary) !important;
}}

/* ── Tabs ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  background: rgba(17,24,39,0.6) !important;
  border-radius: var(--radius-md) !important;
  padding: 4px !important;
  border: 1px solid var(--border-subtle) !important;
  gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent !important;
  color: var(--text-secondary) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 500 !important;
  transition: var(--transition) !important;
  border: none !important;
}}
.stTabs [aria-selected="true"] {{
  background: linear-gradient(135deg, var(--primary), #8B7CF8) !important;
  color: white !important;
  box-shadow: 0 4px 12px var(--glow-primary) !important;
}}

/* ── Expanders ───────────────────────────────────────────────────── */
.streamlit-expanderHeader {{
  background: rgba(17,24,39,0.6) !important;
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border-subtle) !important;
  color: var(--text-primary) !important;
  font-weight: 500 !important;
  transition: var(--transition) !important;
}}
.streamlit-expanderHeader:hover {{
  border-color: rgba(108,99,255,0.4) !important;
  background: rgba(108,99,255,0.1) !important;
}}

/* ── DataFrames ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border-subtle) !important;
  overflow: hidden !important;
}}

/* ── Progress Bars ───────────────────────────────────────────────── */
.stProgress > div > div > div > div {{
  background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
  border-radius: 100px !important;
}}

/* ── Alerts / Info Boxes ─────────────────────────────────────────── */
.stAlert {{
  border-radius: var(--radius-sm) !important;
  border: none !important;
}}

/* ── Divider ─────────────────────────────────────────────────────── */
hr {{ border-color: var(--border-subtle) !important; margin: 1.5rem 0 !important; }}

/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(108,99,255,0.4); border-radius: 100px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--primary); }}

/* ── Animations ──────────────────────────────────────────────────── */
@keyframes fadeInUp {{
  from {{ opacity: 0; transform: translateY(20px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse {{
  0%, 100% {{ box-shadow: 0 0 0 0 var(--glow-primary); }}
  50%       {{ box-shadow: 0 0 0 8px rgba(108,99,255,0); }}
}}
@keyframes shimmer {{
  0%   {{ background-position: -200% center; }}
  100% {{ background-position:  200% center; }}
}}
@keyframes spin {{
  to {{ transform: rotate(360deg); }}
}}

.fade-in-up {{ animation: fadeInUp 0.6s ease forwards; }}
.pulse-glow {{ animation: pulse 2s infinite; }}

/* ── Custom Card Component ───────────────────────────────────────── */
.glass-card {{
  background: linear-gradient(135deg, rgba(17,24,39,0.9), rgba(17,24,39,0.7));
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 1.5rem 2rem;
  backdrop-filter: blur(20px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  transition: var(--transition);
  animation: fadeInUp 0.5s ease;
}}
.glass-card:hover {{
  border-color: rgba(108,99,255,0.35);
  box-shadow: 0 12px 40px var(--glow-primary);
  transform: translateY(-3px);
}}

/* ── Hero Section ────────────────────────────────────────────────── */
.hero-section {{
  background: linear-gradient(135deg, rgba(108,99,255,0.12) 0%, rgba(0,212,170,0.08) 100%);
  border: 1px solid rgba(108,99,255,0.2);
  border-radius: var(--radius-lg);
  padding: 3rem 2.5rem;
  text-align: center;
  backdrop-filter: blur(20px);
  position: relative;
  overflow: hidden;
  animation: fadeInUp 0.8s ease;
}}
.hero-section::before {{
  content: '';
  position: absolute;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: radial-gradient(circle, rgba(108,99,255,0.05) 0%, transparent 60%);
  animation: spin 20s linear infinite;
}}

/* ── Prediction Result Badges ────────────────────────────────────── */
.badge-fake {{
  background: linear-gradient(135deg, rgba(255,71,87,0.2), rgba(255,71,87,0.1));
  border: 1px solid rgba(255,71,87,0.4);
  color: {DANGER_COLOR};
  border-radius: 100px;
  padding: 0.4rem 1.2rem;
  font-weight: 700;
  font-size: 1rem;
  display: inline-block;
  animation: pulse 2s infinite;
}}
.badge-real {{
  background: linear-gradient(135deg, rgba(46,213,115,0.2), rgba(46,213,115,0.1));
  border: 1px solid rgba(46,213,115,0.4);
  color: {SUCCESS_COLOR};
  border-radius: 100px;
  padding: 0.4rem 1.2rem;
  font-weight: 700;
  font-size: 1rem;
  display: inline-block;
}}

/* ── Model Vote Cards ────────────────────────────────────────────── */
.vote-card {{
  border-radius: var(--radius-sm);
  padding: 0.7rem 1rem;
  border: 1px solid var(--border-subtle);
  background: rgba(17,24,39,0.6);
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  transition: var(--transition);
}}
.vote-card:hover {{ border-color: rgba(108,99,255,0.3); transform: translateX(4px); }}
.vote-fake {{ border-left: 3px solid {DANGER_COLOR} !important; }}
.vote-real {{ border-left: 3px solid {SUCCESS_COLOR} !important; }}

/* ── Stat Badge ──────────────────────────────────────────────────── */
.stat-badge {{
  background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(0,212,170,0.1));
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 0.5rem 1rem;
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text-secondary);
  display: inline-block;
  margin: 0.25rem;
}}

/* ── Section Headers ─────────────────────────────────────────────── */
.section-header {{
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}}
.section-header::after {{
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--primary), transparent);
  margin-left: 0.75rem;
}}

/* ── Footer ──────────────────────────────────────────────────────── */
.app-footer {{
  border-top: 1px solid var(--border-subtle);
  padding: 1.5rem 0;
  text-align: center;
  color: var(--text-secondary);
  font-size: 0.8rem;
  margin-top: 3rem;
}}
.app-footer a {{ color: var(--primary); text-decoration: none; transition: var(--transition); }}
.app-footer a:hover {{ color: var(--secondary); }}
</style>
"""


def inject_css() -> None:
    """Inject the global premium CSS into the Streamlit page."""
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def card(content: str, hover: bool = True) -> str:
    """Wrap content in a glassmorphism card div."""
    cls = "glass-card" + (" glass-card-hover" if hover else "")
    return f'<div class="{cls}">{content}</div>'


def badge(text: str, kind: str = "primary") -> str:
    """Return an HTML badge span."""
    colors = {
        "primary": (PRIMARY_COLOR, "rgba(108,99,255,0.2)"),
        "success": (SUCCESS_COLOR, "rgba(46,213,115,0.2)"),
        "danger":  (DANGER_COLOR,  "rgba(255,71,87,0.2)"),
        "warning": (WARNING_COLOR, "rgba(255,165,2,0.2)"),
        "secondary": (SECONDARY_COLOR, "rgba(0,212,170,0.2)"),
    }
    fg, bg = colors.get(kind, colors["primary"])
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {fg}44;'
        f'border-radius:100px;padding:0.25rem 0.75rem;font-size:0.8rem;font-weight:600;">'
        f'{text}</span>'
    )
