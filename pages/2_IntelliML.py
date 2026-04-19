import sys
import os
import streamlit as st

st.set_page_config(
    page_title="IntelliML | Model Recommendation",
    page_icon="🤖",
    layout="wide"
)

# ── Inject intelliml folder into path ────────────────────────────────────────
# Add both the project root and the intelliml folder to ensure imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INTELLIML_PATH = os.path.join(PROJECT_ROOT, 'intelliml')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if INTELLIML_PATH not in sys.path:
    sys.path.insert(0, INTELLIML_PATH)

if "app" in sys.modules:
    del sys.modules["app"]

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .im-header {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1f2d3d 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        .im-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #58a6ff;
            margin: 0;
        }
        .im-subtitle {
            font-size: 1rem;
            color: #8b949e;
            margin-top: 0.3rem;
        }
        .im-badge {
            display: inline-block;
            background: #1f6feb;
            color: white;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
    </style>
    <div class="im-header">
        <p class="im-title">🤖 IntelliML</p>
        <p class="im-subtitle">Automatic Model Recommendation — Let AI pick the best model for your data</p>
        <span class="im-badge">Auto ML</span>
    </div>
""", unsafe_allow_html=True)

# ── Load the IntelliML app ────────────────────────────────────────────────────
try:
    import app as intelliml_app  # intelliml/app.py
except Exception as e:
    st.error(f"❌ Could not load IntelliML app: {e}")
    st.info("Make sure the `intelliml/` folder exists and contains `app.py`")
    st.stop()
