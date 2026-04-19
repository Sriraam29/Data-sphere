import sys
import os
import streamlit as st

st.set_page_config(
    page_title="DataClean | Auto Data Cleaning",
    page_icon="🧹",
    layout="wide"
)

# ── Inject data_clean folder into path ───────────────────────
# Add both the project root and the data_clean folder to ensure imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATACLEAN_PATH = os.path.join(PROJECT_ROOT, 'data_clean')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if DATACLEAN_PATH not in sys.path:
    sys.path.insert(0, DATACLEAN_PATH)

if "app" in sys.modules:
    del sys.modules["app"]

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .dc-header {
            background: linear-gradient(135deg, #0a3d2e 0%, #145a32 50%, #1e8449 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        .dc-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #2ecc71;
            margin: 0;
        }
        .dc-subtitle {
            font-size: 1rem;
            color: #a9dfbf;
            margin-top: 0.3rem;
        }
        .dc-badge {
            display: inline-block;
            background: #27ae60;
            color: white;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
    </style>
    <div class="dc-header">
        <p class="dc-title">🧹 DataClean</p>
        <p class="dc-subtitle">Automatic Data Cleaning — Detect and fix data quality issues instantly</p>
        <span class="dc-badge">Auto Clean</span>
    </div>
""", unsafe_allow_html=True)

# ── Load the DataClean app ────────────────────────────────────────────────────
try:
    import app as dataclean_app  # data_clean/app.py  →  adjust if needed
    if hasattr(dataclean_app, 'main'):
        dataclean_app.main()
except Exception as e:
    st.error(f"❌ Could not load DataClean app: {e}")
    st.info("Make sure the `data_clean/` folder exists and contains `app.py`")
    st.stop()
