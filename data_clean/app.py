"""
╔══════════════════════════════════════════════════════════════════════╗
║        INTELLIGENT DATA CLEANING & ML-AWARE PREPROCESSING APP       ║
║                   Upload → Analyze → Transform → Export             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

import io
import json
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

from logger import TransformationLogger
from schema_inference import infer_schema
from missing_value_handler import handle_missing_values
from outlier_handler import handle_outliers
from encoding_strategy import encode_features
from feature_scaling import scale_features
from feature_selection import select_features
from leakage_detection import detect_leakage
from bias_detection import detect_bias
from drift_detection import detect_drift
from ml_task_detector import detect_task_and_train
from memory_optimizer import optimize_memory
from validator import validate

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
try:
    st.set_page_config(
        page_title="DataForge AI — Intelligent Data Cleaning",
        page_icon="⚗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Industrial Utility Aesthetic
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d0f12;
    --surface: #141720;
    --surface2: #1c2030;
    --border: #2a2f3e;
    --accent: #00e5b0;
    --accent2: #7b61ff;
    --accent3: #ff6b6b;
    --warn: #ffb347;
    --text: #e2e8f0;
    --text-muted: #7a8394;
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', sans-serif;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Cards */
.forge-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

.forge-card-accent {
    background: linear-gradient(135deg, rgba(0,229,176,0.07), rgba(123,97,255,0.07));
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* Hero */
.hero-title {
    font-family: var(--mono);
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -1px;
    line-height: 1.15;
    margin-bottom: 6px;
}
.hero-sub {
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 24px;
}

/* Metric chips */
.metric-chip {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 12px;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 3px;
}
.metric-chip b {
    color: var(--accent);
}

/* Status badges */
.badge-ok {
    background: rgba(0,229,176,0.15);
    color: #00e5b0;
    border: 1px solid rgba(0,229,176,0.4);
    border-radius: 4px;
    padding: 2px 10px;
    font-family: var(--mono);
    font-size: 0.72rem;
}
.badge-warn {
    background: rgba(255,179,71,0.15);
    color: #ffb347;
    border: 1px solid rgba(255,179,71,0.4);
    border-radius: 4px;
    padding: 2px 10px;
    font-family: var(--mono);
    font-size: 0.72rem;
}
.badge-crit {
    background: rgba(255,107,107,0.15);
    color: #ff6b6b;
    border: 1px solid rgba(255,107,107,0.4);
    border-radius: 4px;
    padding: 2px 10px;
    font-family: var(--mono);
    font-size: 0.72rem;
}

/* Step progress */
.step-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--text-muted);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.step-row.active { color: var(--accent); }
.step-row.done   { color: var(--text); }

.step-num {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: var(--surface2);
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    flex-shrink: 0;
}
.step-num.active { background: var(--accent); color: #000; border-color: var(--accent); }
.step-num.done   { background: var(--surface2); color: var(--accent); border-color: var(--accent); }

/* Section headers */
.section-hdr {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 14px;
}

/* Tables */
.stDataFrame thead tr th {
    background: var(--surface2) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
}

/* Input overrides */
.stSelectbox label, .stFileUploader label {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-muted) !important;
}

/* Button */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 4px !important;
    letter-spacing: 0.05em !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    color: var(--text-muted) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def card(content: str, accent: bool = False) -> None:
    cls = "forge-card-accent" if accent else "forge-card"
    st.markdown(f'<div class="{cls}">{content}</div>', unsafe_allow_html=True)


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-hdr">{text}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "ok") -> str:
    return f'<span class="badge-{kind}">{text}</span>'


def chip(label: str, value) -> str:
    return f'<span class="metric-chip">{label}: <b>{value}</b></span>'


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, fname: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        return pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")


@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes, fname: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_parquet(file_bytes: bytes, fname: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(file_bytes))


def load_dataframe(uploaded) -> pd.DataFrame:
    fname = uploaded.name.lower()
    file_bytes = uploaded.read()
    if fname.endswith(".csv"):
        return load_csv(file_bytes, fname)
    elif fname.endswith((".xls", ".xlsx")):
        return load_excel(file_bytes, fname)
    elif fname.endswith(".parquet"):
        return load_parquet(file_bytes, fname)
    else:
        raise ValueError(f"Unsupported file type: {fname}")


def plot_correlation_heatmap(df: pd.DataFrame, title="Correlation Matrix") -> plt.Figure:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    cols = num_df.columns[:25]  # limit
    corr = num_df[cols].corr()
    fig, ax = plt.subplots(figsize=(min(12, len(cols)), min(10, len(cols))), facecolor="#141720")
    ax.set_facecolor("#141720")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        linewidths=0.3,
        linecolor="#1c2030",
        annot=len(cols) <= 12,
        fmt=".2f",
        annot_kws={"size": 7, "color": "#e2e8f0"},
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(title, color="#00e5b0", fontfamily="monospace", fontsize=10, pad=12)
    ax.tick_params(colors="#7a8394", labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_missing_heatmap(df: pd.DataFrame) -> plt.Figure:
    missing = df.isnull()
    if missing.sum().sum() == 0:
        return None
    cols_w_missing = missing.columns[missing.any()].tolist()[:30]
    if not cols_w_missing:
        return None
    fig, ax = plt.subplots(figsize=(12, max(4, len(cols_w_missing) * 0.3)), facecolor="#141720")
    ax.set_facecolor("#141720")
    sns.heatmap(
        missing[cols_w_missing].T,
        ax=ax,
        cbar=False,
        yticklabels=True,
        cmap=["#1c2030", "#ff6b6b"],
    )
    ax.set_title("Missing Value Map (red = missing)", color="#ff6b6b", fontfamily="monospace", fontsize=9)
    ax.tick_params(colors="#7a8394", labelsize=7)
    ax.set_xlabel("Row index", color="#7a8394", fontsize=7)
    fig.tight_layout()
    return fig


def plot_feature_importance(importances: dict, title="Feature Importances", color="#00e5b0") -> plt.Figure:
    if not importances:
        return None
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.35)), facecolor="#141720")
    ax.set_facecolor("#141720")
    bars = ax.barh(labels[::-1], values[::-1], color=color, alpha=0.85, edgecolor="#2a2f3e", linewidth=0.5)
    ax.set_xlabel("Importance", color="#7a8394", fontsize=8)
    ax.set_title(title, color=color, fontfamily="monospace", fontsize=9)
    ax.tick_params(colors="#7a8394", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#2a2f3e")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    return fig


def plot_distributions(df_before: pd.DataFrame, df_after: pd.DataFrame, cols: list, max_plots=6) -> plt.Figure:
    cols = [c for c in cols if c in df_before.columns and c in df_after.columns][:max_plots]
    if not cols:
        return None
    n = len(cols)
    fig, axes = plt.subplots(n, 2, figsize=(10, n * 2.5), facecolor="#141720")
    if n == 1:
        axes = [axes]
    for i, col in enumerate(cols):
        for j, (data, label, color) in enumerate([
            (df_before[col].dropna(), "Before", "#ff6b6b"),
            (df_after[col].dropna() if col in df_after.columns else pd.Series(), "After", "#00e5b0"),
        ]):
            ax = axes[i][j]
            ax.set_facecolor("#141720")
            if len(data) > 0 and pd.api.types.is_numeric_dtype(data):
                ax.hist(data, bins=40, color=color, alpha=0.75, edgecolor="#0d0f12", linewidth=0.3)
            ax.set_title(f"{col} — {label}", color=color, fontfamily="monospace", fontsize=7)
            ax.tick_params(colors="#7a8394", labelsize=6)
            for spine in ax.spines.values():
                spine.set_color("#2a2f3e")
    fig.patch.set_facecolor("#141720")
    fig.tight_layout(pad=1.5)
    return fig


# ─────────────────────────────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_pipeline(
    df_raw: pd.DataFrame,
    target_col: str | None,
    df_drift: pd.DataFrame | None,
    run_ml: bool,
    status_placeholder,
) -> dict:
    """Full cleaning pipeline. Returns results dict."""

    logger = TransformationLogger()
    results: dict = {
        "logger": logger,
        "steps": {},
    }

    def update(msg: str, icon: str = "⚙️"):
        status_placeholder.markdown(
            f'<div style="font-family:monospace;font-size:0.82rem;color:#00e5b0;'
            f'padding:8px 12px;background:#141720;border:1px solid #2a2f3e;border-radius:4px;">'
            f'{icon} {msg}</div>',
            unsafe_allow_html=True,
        )

    # ── STEP 0: Preserve raw copy ─────────────────────────────────────
    df_original = df_raw.copy()
    results["df_raw"] = df_original
    results["shape_before"] = df_raw.shape

    # ── STEP 1: Schema Inference ──────────────────────────────────────
    update("Step 1/9 — Schema inference…", "🔍")
    schema = infer_schema(df_raw, logger)
    results["schema"] = schema

    # Auto-detect target if not specified
    if not target_col:
        target_candidates = [c for c, m in schema.items() if m["is_potential_target"]]
        target_col = target_candidates[0] if target_candidates else None
        if target_col:
            update(f"Step 1/9 — Auto-detected target: '{target_col}'", "🎯")

    results["target_col"] = target_col
    results["steps"]["schema"] = schema

    # ── STEP 2: Bias detection (on raw data, before encoding) ─────────
    update("Step 2/9 — Bias detection…", "⚖️")
    bias_report = detect_bias(df_original, schema, target_col, logger)
    results["steps"]["bias"] = bias_report

    # ── STEP 3: Missing value handling ────────────────────────────────
    update("Step 3/9 — Intelligent missing value imputation…", "🧩")
    df = handle_missing_values(df_raw.copy(), schema, target_col, logger)

    # Update schema for dropped columns
    dropped = [c for c in df_raw.columns if c not in df.columns]
    for c in dropped:
        schema.pop(c, None)

    results["steps"]["missing"] = {
        "dropped_cols": dropped,
        "remaining_missing": int(df.isnull().sum().sum()),
    }

    # ── STEP 4: Outlier handling ──────────────────────────────────────
    update("Step 4/9 — Outlier detection & capping…", "📊")
    df, outlier_report = handle_outliers(df, schema, logger)
    results["steps"]["outliers"] = outlier_report

    # ── STEP 5: Leakage detection ─────────────────────────────────────
    update("Step 5/9 — Data leakage scan…", "🔐")
    leakage_report = detect_leakage(df, schema, target_col, logger)
    results["steps"]["leakage"] = leakage_report

    # ── STEP 6: Encoding ──────────────────────────────────────────────
    update("Step 6/9 — Feature encoding…", "🔢")
    df, encoding_map = encode_features(df, schema, target_col, logger)
    results["encoding_map"] = encoding_map

    # Update schema after encoding (new one-hot columns)
    for col, enc in encoding_map.items():
        if enc.get("method") == "onehot":
            for new_col in enc.get("new_cols", []):
                schema[new_col] = {"inferred_type": "numeric", "is_id_like": False}

    results["steps"]["encoding"] = {
        "methods_used": {col: enc["method"] for col, enc in encoding_map.items()},
    }

    # ── STEP 7: Scaling ───────────────────────────────────────────────
    update("Step 7/9 — Feature scaling…", "📐")
    df, scaler_map = scale_features(df, schema, target_col, encoding_map, logger)
    results["scaler_map"] = scaler_map

    # ── STEP 8: Feature selection ──────────────────────────────────────
    update("Step 8/9 — Feature selection (variance, correlation, MI)…", "✂️")
    df, fs_report = select_features(df, target_col, logger)
    results["steps"]["feature_selection"] = fs_report

    # ── Memory optimization ────────────────────────────────────────────
    df, mem_report = optimize_memory(df, schema, logger)
    results["steps"]["memory"] = mem_report

    # ── Validation ────────────────────────────────────────────────────
    val_report, df = validate(df, logger)
    results["steps"]["validation"] = val_report

    results["df_clean"] = df
    results["shape_after"] = df.shape

    # ── STEP 9: ML task detection ─────────────────────────────────────
    if run_ml and target_col and target_col in df.columns:
        update("Step 9/9 — Training baseline model & computing SHAP…", "🤖")
        ml_report = detect_task_and_train(df, target_col, logger)
        results["steps"]["ml"] = ml_report
    else:
        results["steps"]["ml"] = {}

    # ── Drift detection ────────────────────────────────────────────────
    if df_drift is not None:
        update("Bonus — Data drift analysis…", "🌊")
        drift_report = detect_drift(df_original, df_drift, schema, logger)
        results["steps"]["drift"] = drift_report

    update("✅ Pipeline complete!", "✅")
    return results


# ─────────────────────────────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────────────────────────────

def build_pipeline_object(encoding_map: dict, scaler_map: dict) -> dict:
    return {"encoding_map": encoding_map, "scaler_map": scaler_map}


def generate_html_report(results: dict) -> str:
    schema = results.get("schema", {})
    steps = results.get("steps", {})
    shape_before = results.get("shape_before", ())
    shape_after = results.get("shape_after", ())
    log_entries = results["logger"].entries
    target_col = results.get("target_col", "N/A")

    warnings = [e for e in log_entries if e["severity"] == "warning"]
    criticals = [e for e in log_entries if e["severity"] == "critical"]

    # Feature importances
    ml = steps.get("ml", {})
    fi = ml.get("feature_importances", {})
    top_fi = list(fi.items())[:10]

    fi_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.5f}</td></tr>" for k, v in top_fi
    )

    warn_rows = "".join(
        f'<tr style="color:#ffb347"><td>{e["ts"]}</td><td>{e["stage"]}</td>'
        f'<td>{e["action"]}</td><td>{e["column"] or ""}</td><td>{e["reason"]}</td></tr>'
        for e in warnings
    )
    crit_rows = "".join(
        f'<tr style="color:#ff6b6b"><td>{e["ts"]}</td><td>{e["stage"]}</td>'
        f'<td>{e["action"]}</td><td>{e["column"] or ""}</td><td>{e["reason"]}</td></tr>'
        for e in criticals
    )

    mem = steps.get("memory", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DataForge AI — Cleaning Report</title>
<style>
body {{ font-family: 'Courier New', monospace; background: #0d0f12; color: #e2e8f0; padding: 40px; }}
h1 {{ color: #00e5b0; font-size: 2rem; border-bottom: 2px solid #00e5b0; padding-bottom: 12px; }}
h2 {{ color: #7b61ff; font-size: 1.1rem; margin-top: 32px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th {{ background: #1c2030; color: #00e5b0; padding: 8px 12px; text-align: left; font-size: 0.8rem; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #2a2f3e; font-size: 0.78rem; }}
.chip {{ display: inline-block; background: #1c2030; border: 1px solid #2a2f3e;
         border-radius: 4px; padding: 3px 10px; margin: 3px; font-size: 0.75rem; }}
.chip b {{ color: #00e5b0; }}
.warn {{ color: #ffb347; }}
.crit {{ color: #ff6b6b; }}
</style>
</head>
<body>
<h1>⚗️ DataForge AI — Cleaning Report</h1>

<h2>📋 Summary</h2>
<span class="chip">Rows before: <b>{shape_before[0]}</b></span>
<span class="chip">Cols before: <b>{shape_before[1]}</b></span>
<span class="chip">Rows after: <b>{shape_after[0]}</b></span>
<span class="chip">Cols after: <b>{shape_after[1]}</b></span>
<span class="chip">Target: <b>{target_col}</b></span>
<span class="chip">Memory before: <b>{mem.get("before_mb", "?")} MB</b></span>
<span class="chip">Memory after: <b>{mem.get("after_mb", "?")} MB</b></span>
<span class="chip">Memory saved: <b>{mem.get("reduction_pct", "?")}%</b></span>

<h2>🔍 Schema Inference</h2>
<table>
<tr>
  <th>Column</th><th>Inferred Type</th><th>Missing Rate</th>
  <th>Cardinality</th><th>ID-like</th><th>Target Risk</th><th>Leakage Risk</th>
</tr>
{"".join(
    f'<tr><td>{col}</td><td>{m["inferred_type"]}</td><td>{m["missing_rate"]:.2%}</td>'
    f'<td>{m["cardinality"]}</td><td>{m["is_id_like"]}</td>'
    f'<td>{"✓" if m["is_potential_target"] else ""}</td>'
    f'<td>{"⚠️" if m["is_leakage_risk"] else ""}</td></tr>'
    for col, m in schema.items()
)}
</table>

<h2>⚠️ Warnings</h2>
<table>
<tr><th>Timestamp</th><th>Stage</th><th>Action</th><th>Column</th><th>Reason</th></tr>
{warn_rows if warn_rows else "<tr><td colspan='5'>No warnings.</td></tr>"}
</table>

<h2>🚨 Critical Issues</h2>
<table>
<tr><th>Timestamp</th><th>Stage</th><th>Action</th><th>Column</th><th>Reason</th></tr>
{crit_rows if crit_rows else "<tr><td colspan='5'>No critical issues.</td></tr>"}
</table>

<h2>🤖 ML Baseline</h2>
<span class="chip">Task: <b>{ml.get("task", "N/A")}</b></span>
<span class="chip">Model: <b>{ml.get("model_used", "N/A")}</b></span>
<span class="chip">Score: <b>{ml.get("model_score", "N/A")}</b></span>
<span class="chip">Metric: <b>{ml.get("scoring_metric", "N/A")}</b></span>

<h2>📊 Top Feature Importances</h2>
<table>
<tr><th>Feature</th><th>Importance</th></tr>
{fi_rows if fi_rows else "<tr><td colspan='2'>Not available.</td></tr>"}
</table>

<h2>📝 Full Transformation Log</h2>
<table>
<tr><th>Timestamp</th><th>Stage</th><th>Action</th><th>Column</th><th>Severity</th><th>Reason</th></tr>
{"".join(
    f'<tr><td>{e["ts"]}</td><td>{e["stage"]}</td><td>{e["action"]}</td>'
    f'<td>{e["column"] or ""}</td><td>{e["severity"]}</td><td>{e["reason"]}</td></tr>'
    for e in log_entries
)}
</table>
</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="font-family:monospace;font-size:1.4rem;font-weight:700;'
            'color:#00e5b0;margin-bottom:4px;">⚗️ DataForge AI</div>'
            '<div style="font-family:monospace;font-size:0.65rem;color:#7a8394;'
            'letter-spacing:0.15em;margin-bottom:20px;">INTELLIGENT PREPROCESSING ENGINE</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-hdr">PRIMARY DATASET</div>', unsafe_allow_html=True)
        uploaded_main = st.file_uploader(
            "Upload dataset", type=["csv", "xlsx", "parquet"], key="main_upload"
        )

        st.markdown('<div class="section-hdr">DRIFT DATASET (Optional)</div>', unsafe_allow_html=True)
        uploaded_drift = st.file_uploader(
            "Upload comparison dataset", type=["csv", "xlsx", "parquet"], key="drift_upload"
        )

        st.markdown('<div class="section-hdr">CONFIGURATION</div>', unsafe_allow_html=True)
        target_input = st.text_input(
            "Target column (leave blank for auto-detect)",
            value="",
            help="The column you want to predict. Leave empty for auto-detection.",
        )
        run_ml = st.checkbox("Run baseline ML model + SHAP", value=True)
        run_btn = st.button("🚀 RUN PIPELINE", use_container_width=True)

        st.markdown('<div class="section-hdr">PIPELINE STEPS</div>', unsafe_allow_html=True)
        steps_html = """
        <div class="step-row"><div class="step-num">1</div> Schema Inference</div>
        <div class="step-row"><div class="step-num">2</div> Bias Detection</div>
        <div class="step-row"><div class="step-num">3</div> Missing Values</div>
        <div class="step-row"><div class="step-num">4</div> Outlier Handling</div>
        <div class="step-row"><div class="step-num">5</div> Leakage Detection</div>
        <div class="step-row"><div class="step-num">6</div> Feature Encoding</div>
        <div class="step-row"><div class="step-num">7</div> Feature Scaling</div>
        <div class="step-row"><div class="step-num">8</div> Feature Selection</div>
        <div class="step-row"><div class="step-num">9</div> ML Baseline + SHAP</div>
        """
        st.markdown(steps_html, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero-title">DataForge AI</div>'
        '<div class="hero-sub">INTELLIGENT DATA CLEANING & ML-AWARE PREPROCESSING ENGINE</div>',
        unsafe_allow_html=True,
    )

    if not uploaded_main:
        # ── Welcome screen ─────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            card(
                "<div style='font-family:monospace;font-size:0.75rem;color:#7a8394;'>"
                "UPLOAD CHAOS</div>"
                "<div style='font-size:2rem;margin:8px 0;'>📂</div>"
                "<div style='font-size:0.85rem;'>Drop any messy CSV, Excel, or Parquet file.</div>",
            )
        with col2:
            card(
                "<div style='font-family:monospace;font-size:0.75rem;color:#7a8394;'>"
                "AI ANALYZES</div>"
                "<div style='font-size:2rem;margin:8px 0;'>🧠</div>"
                "<div style='font-size:0.85rem;'>9-stage automated pipeline. "
                "Schema → Imputation → Outliers → Encoding → Selection.</div>",
                accent=True,
            )
        with col3:
            card(
                "<div style='font-family:monospace;font-size:0.75rem;color:#7a8394;'>"
                "RECEIVE INTELLIGENCE</div>"
                "<div style='font-size:2rem;margin:8px 0;'>⚡</div>"
                "<div style='font-size:0.85rem;'>Clean CSV/Parquet + JSON log + "
                "ML pipeline + HTML report.</div>",
            )

        st.markdown(
            '<div style="font-family:monospace;font-size:0.75rem;color:#7a8394;text-align:center;'
            'margin-top:32px;">Supports up to 1M rows · Statistically grounded decisions · '
            'Zero configuration required</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Load data ─────────────────────────────────────────────────────
    try:
        df_raw = load_dataframe(uploaded_main)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return

    df_drift = None
    if uploaded_drift:
        try:
            df_drift = load_dataframe(uploaded_drift)
        except Exception as e:
            st.warning(f"Could not load drift dataset: {e}")

    target_col = target_input.strip() if target_input.strip() else None
    if target_col and target_col not in df_raw.columns:
        st.error(f"Target column '{target_col}' not found in dataset.")
        target_col = None

    # ── Raw Data Preview ──────────────────────────────────────────────
    section_header("RAW DATA PREVIEW")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df_raw.shape[0]:,}")
    col2.metric("Columns", df_raw.shape[1])
    col3.metric("Missing cells", f"{df_raw.isnull().sum().sum():,}")
    col4.metric("Duplicate rows", f"{df_raw.duplicated().sum():,}")

    with st.expander("📋 Raw data sample (first 100 rows)", expanded=False):
        st.dataframe(df_raw.head(100), use_container_width=True)

    with st.expander("📊 Raw statistics", expanded=False):
        st.dataframe(df_raw.describe(include="all").T, use_container_width=True)

    # Missing heatmap on raw
    fig_miss = plot_missing_heatmap(df_raw)
    if fig_miss:
        with st.expander("🗺️ Missing value map (raw)", expanded=False):
            st.pyplot(fig_miss, clear_figure=True)
            plt.close()

    # ── Run Pipeline ──────────────────────────────────────────────────
    if run_btn or "results" in st.session_state:
        if run_btn:
            status_ph = st.empty()
            try:
                with st.spinner(""):
                    results = run_pipeline(
                        df_raw=df_raw,
                        target_col=target_col,
                        df_drift=df_drift,
                        run_ml=run_ml,
                        status_placeholder=status_ph,
                    )
                st.session_state["results"] = results
                status_ph.empty()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.code(traceback.format_exc())
                return

        results = st.session_state.get("results")
        if not results:
            return

        df_clean = results["df_clean"]
        steps = results["steps"]
        schema = results["schema"]
        logger: TransformationLogger = results["logger"]
        shape_before = results["shape_before"]
        shape_after = results["shape_after"]
        detected_target = results.get("target_col")

        # ── Success banner ────────────────────────────────────────────
        rows_removed = shape_before[0] - shape_after[0]
        cols_removed = shape_before[1] - shape_after[1]
        mem = steps.get("memory", {})

        st.success(
            f"✅ Pipeline complete — "
            f"{shape_after[0]:,} rows × {shape_after[1]} cols | "
            f"{mem.get('reduction_pct', 0):.1f}% memory reduction | "
            f"target: **{detected_target or 'none detected'}**"
        )

        # ── Metrics overview ──────────────────────────────────────────
        section_header("BEFORE vs AFTER")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", f"{shape_after[0]:,}", delta=f"{-rows_removed:+,}" if rows_removed else "0")
        m2.metric("Columns", shape_after[1], delta=f"{-cols_removed:+,}" if cols_removed else "0")
        m3.metric(
            "Missing cells",
            f"{df_clean.isnull().sum().sum():,}",
            delta=f"-{df_raw.isnull().sum().sum():,}",
        )
        m4.metric("Memory", f"{mem.get('after_mb', 0):.1f} MB", delta=f"-{mem.get('reduction_pct', 0):.1f}%")
        val = steps.get("validation", {})
        m5.metric("Validation", "✅ PASS" if val.get("overall_passed") else "⚠️ WARN")

        # ── Clean data preview ────────────────────────────────────────
        section_header("CLEAN DATASET PREVIEW")
        with st.expander("📋 Clean data sample (first 100 rows)", expanded=True):
            st.dataframe(df_clean.head(100), use_container_width=True)

        # ── Schema report ─────────────────────────────────────────────
        section_header("SCHEMA INFERENCE REPORT")
        schema_rows = []
        for col, m in schema.items():
            schema_rows.append({
                "Column": col,
                "Type": m.get("inferred_type", "?"),
                "Missing Rate": f"{m.get('missing_rate', 0):.2%}",
                "Cardinality": m.get("cardinality", "?"),
                "Entropy": m.get("entropy", 0),
                "ID-like": "⚠️" if m.get("is_id_like") else "",
                "Target Signal": "🎯" if m.get("is_potential_target") else "",
                "Leakage Risk": "🔐" if m.get("is_leakage_risk") else "",
            })
        if schema_rows:
            st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

        # ── Tabs for detailed results ─────────────────────────────────
        tabs = st.tabs([
            "📊 Visualizations",
            "🧩 Missing & Outliers",
            "⚖️ Bias Detection",
            "🔐 Leakage Detection",
            "🤖 ML Baseline",
            "🌊 Drift Analysis",
            "📝 Transform Log",
            "💾 Export",
        ])

        # ── Tab 0: Visualizations ─────────────────────────────────────
        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Correlation Matrix (Clean)**")
                fig = plot_correlation_heatmap(df_clean, "After Cleaning")
                if fig:
                    st.pyplot(fig, clear_figure=True)
                    plt.close()
                else:
                    st.info("Not enough numeric columns for correlation matrix.")

            with c2:
                st.markdown("**Correlation Matrix (Raw)**")
                fig2 = plot_correlation_heatmap(df_raw, "Before Cleaning")
                if fig2:
                    st.pyplot(fig2, clear_figure=True)
                    plt.close()
                else:
                    st.info("Not enough numeric columns.")

            # Distribution before/after
            num_cols_raw = list(df_raw.select_dtypes(include=[np.number]).columns[:6])
            if num_cols_raw:
                st.markdown("**Distribution Shift (Before vs After)**")
                fig_dist = plot_distributions(df_raw, df_clean, num_cols_raw)
                if fig_dist:
                    st.pyplot(fig_dist, clear_figure=True)
                    plt.close()

        # ── Tab 1: Missing & Outliers ─────────────────────────────────
        with tabs[1]:
            miss_info = steps.get("missing", {})
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Missing Value Handling**")
                st.markdown(
                    f"Dropped columns: **{miss_info.get('dropped_cols', [])}**  \n"
                    f"Remaining missing after imputation: **{miss_info.get('remaining_missing', 0)}**"
                )
            with c2:
                st.markdown("**Outlier Report**")
                outlier_data = steps.get("outliers", {})
                if outlier_data:
                    o_rows = [
                        {
                            "Column": col,
                            "Method": v["method"],
                            "N outliers": v["n_outliers"],
                            "Rate": f"{v['outlier_rate']:.2%}",
                            "Skewness": v["skewness"],
                            "Kurtosis": v["kurtosis"],
                        }
                        for col, v in outlier_data.items()
                    ]
                    st.dataframe(pd.DataFrame(o_rows), use_container_width=True)
                else:
                    st.info("No outlier processing applied.")

        # ── Tab 2: Bias ───────────────────────────────────────────────
        with tabs[2]:
            bias_data = steps.get("bias", {})
            if bias_data:
                for col, b in bias_data.items():
                    is_b = b.get("is_biased", False)
                    sig = b.get("statistically_significant", False)
                    level = "crit" if (is_b and sig) else ("warn" if is_b else "ok")
                    status = "⚠️ BIASED" if is_b else "✅ OK"
                    with st.expander(f"{badge(status, level)} {col} — disparity={b['disparity']:.4f}, p={b.get('p_value', 'N/A')}", expanded=is_b):
                        gs = b.get("group_stats", {})
                        if gs:
                            st.dataframe(pd.DataFrame(gs).T, use_container_width=True)
            else:
                st.info(
                    "No sensitive attributes detected. Bias detection triggers on columns matching: "
                    "gender, race, age, income, religion, etc."
                )

        # ── Tab 3: Leakage ────────────────────────────────────────────
        with tabs[3]:
            leak_data = steps.get("leakage", {})
            if leak_data:
                for col, l in leak_data.items():
                    flags = l.get("flags", [])
                    corr = l.get("correlation_to_target")
                    is_crit = any("corr_to_target" in f for f in flags)
                    lvl = "crit" if is_crit else "warn"
                    with st.expander(
                        f"{badge('CRITICAL LEAK' if is_crit else 'RISK', lvl)} {col}",
                        expanded=is_crit,
                    ):
                        st.markdown(f"**Flags:** {', '.join(flags)}")
                        if corr is not None:
                            st.markdown(f"**Correlation to target:** `{corr}`")
            else:
                st.info("✅ No leakage detected.")

        # ── Tab 4: ML ─────────────────────────────────────────────────
        with tabs[4]:
            ml = steps.get("ml", {})
            if ml and not ml.get("error"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Task", ml.get("task", "N/A"))
                c2.metric("Model", ml.get("model_used", "N/A"))
                c3.metric(ml.get("scoring_metric", "Score"), ml.get("model_score", "N/A"))
                c4.metric("Features", ml.get("n_features", "N/A"))

                c1, c2 = st.columns(2)
                with c1:
                    fi = ml.get("feature_importances", {})
                    if fi:
                        fig_fi = plot_feature_importance(fi, "Feature Importances", "#00e5b0")
                        if fig_fi:
                            st.pyplot(fig_fi, clear_figure=True)
                            plt.close()
                with c2:
                    shap = ml.get("shap_values", {})
                    if shap:
                        fig_sh = plot_feature_importance(shap, "SHAP Values (Mean |SHAP|)", "#7b61ff")
                        if fig_sh:
                            st.pyplot(fig_sh, clear_figure=True)
                            plt.close()
            elif ml.get("error"):
                st.warning(f"ML baseline: {ml['error']}")
            else:
                st.info("ML baseline was not run. Enable it in the sidebar.")

        # ── Tab 5: Drift ──────────────────────────────────────────────
        with tabs[5]:
            drift_data = steps.get("drift", {})
            if drift_data:
                summary = drift_data.get("__summary__", {})
                st.markdown(
                    f"**Columns tested:** {summary.get('total_columns_tested', 0)} | "
                    f"**Drifted:** {summary.get('drifted_columns', 0)} | "
                    f"**Drift rate:** {summary.get('drift_rate', 0):.2%}"
                )
                drift_rows = [
                    {
                        "Column": col,
                        "Test": v.get("test"),
                        "Statistic": v.get("statistic"),
                        "p-value": v.get("p_value"),
                        "Drifted": "🚨 YES" if v.get("drifted") else "✅ No",
                    }
                    for col, v in drift_data.items()
                    if col != "__summary__"
                ]
                if drift_rows:
                    st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)
            else:
                st.info("Upload a second dataset in the sidebar to enable drift detection.")

        # ── Tab 6: Log ────────────────────────────────────────────────
        with tabs[6]:
            log_summary = logger.summary()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total actions", log_summary["total_actions"])
            col2.metric("Warnings", log_summary["warnings"])
            col3.metric("Critical", log_summary["critical"])

            entries_df = pd.DataFrame(logger.entries)
            if not entries_df.empty:
                severity_filter = st.multiselect(
                    "Filter by severity",
                    options=["info", "warning", "critical"],
                    default=["warning", "critical"],
                )
                filtered = entries_df[entries_df["severity"].isin(severity_filter)]
                st.dataframe(filtered, use_container_width=True)

        # ── Tab 7: Export ─────────────────────────────────────────────
        with tabs[7]:
            section_header("DOWNLOAD ARTIFACTS")

            c1, c2 = st.columns(2)

            # CSV
            with c1:
                csv_bytes = df_clean.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Clean CSV",
                    data=csv_bytes,
                    file_name="clean_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # Parquet
            with c2:
                try:
                    pq_buf = io.BytesIO()
                    df_clean.to_parquet(pq_buf, index=False)
                    st.download_button(
                        "⬇️ Download Clean Parquet",
                        data=pq_buf.getvalue(),
                        file_name="clean_dataset.parquet",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
                except Exception:
                    st.info("Parquet export unavailable (pyarrow not installed).")

            # JSON log
            with c1:
                st.download_button(
                    "⬇️ Download Transformation Log (JSON)",
                    data=logger.to_json().encode(),
                    file_name="cleaning_log.json",
                    mime="application/json",
                    use_container_width=True,
                )

            # HTML Report
            with c2:
                html_report = generate_html_report(results)
                st.download_button(
                    "⬇️ Download HTML Report",
                    data=html_report.encode(),
                    file_name="cleaning_report.html",
                    mime="text/html",
                    use_container_width=True,
                )

            # Pipeline (joblib)
            pipeline_obj = build_pipeline_object(
                results.get("encoding_map", {}),
                results.get("scaler_map", {}),
            )
            pipeline_buf = io.BytesIO()
            joblib.dump(pipeline_obj, pipeline_buf)
            with c1:
                st.download_button(
                    "⬇️ Download ML Pipeline (.joblib)",
                    data=pipeline_buf.getvalue(),
                    file_name="preprocessing_pipeline.joblib",
                    mime="application/octet-stream",
                    use_container_width=True,
                )

            # Full schema JSON
            with c2:
                st.download_button(
                    "⬇️ Download Schema (JSON)",
                    data=json.dumps(results.get("schema", {}), indent=2, default=str).encode(),
                    file_name="schema.json",
                    mime="application/json",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
