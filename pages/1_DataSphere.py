import sys
import os
import streamlit as st

st.set_page_config(
    page_title="DataSphere | NLP to SQL",
    page_icon="🗄️",
    layout="wide"
)

# ── Inject DataSphere folder into path ──────────────────────────────────────
DATASPHERE_PATH = os.path.join(os.path.dirname(__file__), '..', 'DataSphere')
sys.path.insert(0, os.path.abspath(DATASPHERE_PATH))

if "app" in sys.modules:
    del sys.modules["app"]

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .ds-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        .ds-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #e94560;
            margin: 0;
        }
        .ds-subtitle {
            font-size: 1rem;
            color: #a8b2d8;
            margin-top: 0.3rem;
        }
        .ds-badge {
            display: inline-block;
            background: #e94560;
            color: white;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
    </style>
    <div class="ds-header">
        <p class="ds-title">🗄️ DataSphere</p>
        <p class="ds-subtitle">Natural Language to SQL — Query your database in plain English</p>
        <span class="ds-badge">NLP → SQL</span>
    </div>
""", unsafe_allow_html=True)

# ── Load the DataSphere app ───────────────────────────────────────────────────
try:
    import app as datasphere_app  # DataSphere/app.py
except Exception as e:
    st.error(f"❌ Could not load DataSphere app: {e}")
    st.info("Make sure the `DataSphere/` folder exists and contains `app.py`")
    st.stop()
