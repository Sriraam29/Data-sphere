"""
IntelliML — Intelligent Supervised & Unsupervised Model Selection System
Streamlit Frontend · scikit-learn · XGBoost · LightGBM · SHAP · Optuna · UMAP
"""

import io
import os
import warnings
import time
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Page config (must be first) ──────────────────────────────────────────────
st.set_page_config(
    page_title="IntelliML — Model Selection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Import ML engine ─────────────────────────────────────────────────────────
# Use a relative import when running inside the `intelliml` package
from .ml_engine import (
    profile_dataset, detect_task, auto_select_metric,
    prepare_supervised_data, metric_label,
    run_parallel_cv, statistical_comparison,
    train_final_model, extract_feature_importance,
    compute_shap, compute_classification_metrics, compute_regression_metrics,
    run_unsupervised_pipeline,
    model_to_bytes, results_to_leaderboard_csv, results_to_json,
    charts,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --bg:        #050c18;
  --surface:   #0a1628;
  --card:      #0f1e35;
  --border:    #1a3050;
  --accent:    #00d4ff;
  --accent2:   #7c3aed;
  --success:   #10d98a;
  --warn:      #f59e0b;
  --danger:    #ef4444;
  --text:      #e2e8f0;
  --muted:     #64748b;
}

html, body, [class*="css"] {
  font-family: 'Syne', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Main content */
.main .block-container { padding-top: 1.5rem; max-width: 1400px; }

/* Metric cards */
[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
}
[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 11px !important; letter-spacing: 1px; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.6rem !important; }
[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; }

/* Dataframes */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600;
  color: var(--muted) !important;
  border-bottom: 2px solid transparent;
  font-size: 13px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  letter-spacing: 0.5px;
  transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Select / radio */
.stSelectbox > div, .stRadio > div { font-family: 'Syne', sans-serif !important; }
.stSelectbox [data-baseweb="select"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
.stSelectbox [data-baseweb="select"] * { color: var(--text) !important; }

/* Sliders */
.stSlider [data-baseweb="slider"] { color: var(--accent) !important; }

/* Expanders */
[data-testid="stExpander"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
}

/* Progress */
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

/* Info / success / warning / error boxes */
.stAlert { border-radius: 8px !important; font-family: 'Syne', sans-serif !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Code blocks */
code { font-family: 'JetBrains Mono', monospace !important; background: var(--surface) !important;
  color: var(--accent) !important; padding: 2px 6px; border-radius: 4px; }

/* File uploader */
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 10px !important;
}

/* Badges via markdown */
.badge-cyan  { background:rgba(0,212,255,0.12); color:#00d4ff; border:1px solid rgba(0,212,255,0.3); border-radius:20px; padding:2px 10px; font-size:11px; font-family:'JetBrains Mono',monospace; display:inline-block; }
.badge-green { background:rgba(16,217,138,0.12); color:#10d98a; border:1px solid rgba(16,217,138,0.3); border-radius:20px; padding:2px 10px; font-size:11px; font-family:'JetBrains Mono',monospace; display:inline-block; }
.badge-amber { background:rgba(245,158,11,0.12); color:#f59e0b; border:1px solid rgba(245,158,11,0.3); border-radius:20px; padding:2px 10px; font-size:11px; font-family:'JetBrains Mono',monospace; display:inline-block; }
.badge-red   { background:rgba(239,68,68,0.12); color:#ef4444; border:1px solid rgba(239,68,68,0.3); border-radius:20px; padding:2px 10px; font-size:11px; font-family:'JetBrains Mono',monospace; display:inline-block; }
.badge-purple{ background:rgba(124,58,237,0.12); color:#a78bfa; border:1px solid rgba(124,58,237,0.3); border-radius:20px; padding:2px 10px; font-size:11px; font-family:'JetBrains Mono',monospace; display:inline-block; }

/* Recommendation box */
.rec-box {
  background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(124,58,237,0.06));
  border: 1px solid rgba(0,212,255,0.3);
  border-radius: 12px;
  padding: 20px 24px;
  margin: 16px 0;
}

/* Header */
.header-bar {
  display: flex; align-items: center; gap: 14px;
  padding: 14px 0; border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
}
.logo-box {
  width: 40px; height: 40px;
  background: linear-gradient(135deg, #00d4ff, #7c3aed);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 16px; color: white;
}
.formula-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--muted);
}

/* Plotly chart backgrounds */
.js-plotly-plot { border: 1px solid var(--border); border-radius: 10px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <div class="logo-box">ML</div>
  <div>
    <div style="font-size:22px; font-weight:800; letter-spacing:0.5px;">IntelliML</div>
    <div style="font-size:10px; color:var(--muted); letter-spacing:2px; font-family:'JetBrains Mono',monospace;">
      INTELLIGENT MODEL SELECTION · VALIDATION · DISCOVERY SYSTEM
    </div>
  </div>
  <div style="margin-left:auto;">
    <span class="badge-cyan">scikit-learn</span>&nbsp;
    <span class="badge-purple">XGBoost · LightGBM</span>&nbsp;
    <span class="badge-green">SHAP · Optuna</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("df", None), ("profile", None), ("sup_results", None),
    ("unsup_results", None), ("unsup_df_out", None),
    ("final_pipeline", None), ("X_test", None), ("y_test", None),
    ("y_pred", None), ("y_pred_proba", None),
    ("clf_metrics", None), ("reg_metrics", None),
    ("fi_df", None), ("shap_vals", None), ("shap_X_tr", None),
    ("mode", "supervised"), ("ran_analysis", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
#  SAMPLE DATA
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_CLASSIFICATION = """age,income,education_years,credit_score,debt_ratio,employment_years,loan_approved
35,75000,16,720,0.28,8,1
28,45000,14,650,0.42,3,0
52,120000,18,780,0.18,22,1
24,32000,12,580,0.61,1,0
41,89000,16,710,0.31,14,1
38,67000,15,695,0.38,11,1
29,38000,13,620,0.55,4,0
55,145000,20,810,0.15,28,1
31,52000,14,660,0.44,6,0
47,98000,17,745,0.25,18,1
22,28000,12,550,0.72,0,0
60,180000,18,820,0.12,35,1
33,61000,15,680,0.35,7,1
44,105000,16,755,0.22,16,1
26,41000,13,600,0.58,2,0
39,72000,15,700,0.33,12,1
51,135000,18,790,0.17,24,1
27,35000,12,590,0.63,1,0
43,92000,17,730,0.27,15,1
36,68000,15,690,0.37,9,1
30,49000,14,640,0.46,5,0
58,165000,20,815,0.13,32,1
25,33000,12,565,0.68,1,0
48,110000,17,760,0.21,20,1
34,63000,15,685,0.36,8,1
42,88000,16,720,0.29,14,1
23,29000,12,545,0.74,0,0
56,150000,19,800,0.16,30,1
37,70000,15,698,0.34,10,1
45,100000,17,748,0.24,17,1
40,85000,16,715,0.30,13,1
32,55000,14,665,0.41,6,0
49,115000,17,765,0.20,21,1
21,26000,11,530,0.78,0,0
53,140000,19,795,0.16,25,1"""

SAMPLE_REGRESSION = """sqft,bedrooms,bathrooms,age_years,garage_spaces,lot_size,neighborhood_score,price
1200,2,1,15,0,4500,6.2,185000
1850,3,2,8,1,6200,7.8,285000
2400,4,2,3,2,8100,8.5,420000
980,1,1,25,0,3800,5.1,145000
1650,3,2,12,1,5800,7.2,255000
2100,4,3,5,2,7500,8.8,380000
1400,2,2,18,1,5000,6.8,210000
3200,5,3,2,2,10200,9.2,580000
1100,2,1,22,0,4100,5.8,165000
1750,3,2,9,1,6000,7.5,270000
2600,4,3,4,2,8800,9.0,450000
1050,2,1,28,0,3900,4.9,155000
1900,3,2,7,1,6500,7.9,295000
2200,4,3,6,2,7800,8.9,395000
1300,2,2,16,1,4700,6.5,200000
3500,5,4,1,3,11500,9.5,650000
1150,2,1,20,0,4200,5.6,172000
1800,3,2,10,1,6100,7.6,278000
2500,4,3,3,2,8500,9.1,435000
1000,1,1,30,0,3700,4.7,148000
1700,3,2,11,1,5900,7.3,262000
2300,4,3,5,2,7900,8.7,408000
1250,2,2,17,1,4600,6.3,192000
3800,5,4,2,3,12200,9.6,698000
1350,2,2,14,1,4900,6.6,208000
2450,4,3,4,2,8200,8.9,428000
1600,3,2,13,1,5600,7.0,248000
2800,5,3,3,2,9500,9.3,510000"""

SAMPLE_CLUSTERING = """annual_spend,visit_frequency,avg_order_value,loyalty_score,returns_rate,tenure_months,nps_score
12500,45,278,8.2,0.05,36,72
3200,12,267,4.1,0.18,8,42
28000,78,359,9.5,0.02,60,91
8500,28,304,6.8,0.09,24,65
1800,8,225,3.2,0.25,4,31
22000,62,355,9.1,0.03,52,88
5500,20,275,5.4,0.14,15,55
35000,95,368,9.8,0.01,72,95
9800,32,306,7.2,0.08,28,68
2400,10,240,3.8,0.22,6,38
18000,55,327,8.7,0.04,44,84
6800,24,283,5.9,0.12,18,58
42000,108,389,9.9,0.01,84,97
11200,38,295,7.5,0.07,32,71
1500,6,250,2.9,0.28,3,28
25000,70,357,9.3,0.02,56,90
7200,26,277,6.1,0.11,20,60
4100,15,273,4.5,0.20,10,45
31000,85,365,9.6,0.02,68,94
8900,30,297,7.0,0.09,26,67
2100,9,233,3.5,0.24,5,35
19500,58,334,8.9,0.03,47,86
6200,22,282,5.7,0.13,17,57
14000,48,292,8.0,0.06,38,74
3800,14,271,4.3,0.19,9,43
27500,75,361,9.4,0.02,58,92
10500,36,300,7.3,0.08,30,69
4800,18,270,5.1,0.16,13,52"""


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()

    # ── Analysis Mode ──
    st.markdown("**Analysis Mode**")
    mode = st.radio(
        "mode", ["supervised", "unsupervised"],
        format_func=lambda x: "⚡ Supervised" if x == "supervised" else "🔍 Unsupervised",
        horizontal=True, label_visibility="collapsed",
    )
    st.session_state["mode"] = mode
    st.divider()

    # ── Data Source ──
    st.markdown("**Data Source**")
    data_source = st.radio(
        "data_src", ["Upload CSV", "Sample Dataset"],
        horizontal=True, label_visibility="collapsed"
    )

    df = None
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"✓ Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
            except Exception as e:
                st.error(f"Parse error: {e}")
    else:
        sample_choice = st.selectbox(
            "Sample", ["Binary Classification", "Regression", "Clustering"],
            label_visibility="collapsed"
        )
        samples = {
            "Binary Classification": SAMPLE_CLASSIFICATION,
            "Regression": SAMPLE_REGRESSION,
            "Clustering": SAMPLE_CLUSTERING,
        }
        df = pd.read_csv(io.StringIO(samples[sample_choice]))
        st.info(f"Sample: {df.shape[0]} rows × {df.shape[1]} cols")

    if df is not None:
        st.session_state["df"] = df
    st.divider()

    # ── Supervised Config ──
    if mode == "supervised" and st.session_state["df"] is not None:
        st.markdown("**Supervised Settings**")
        df_tmp = st.session_state["df"]

        target_col = st.selectbox("🎯 Target Column", df_tmp.columns.tolist(),
                                  index=len(df_tmp.columns) - 1)
        task_override = st.selectbox(
            "Task Type", ["Auto Detect", "Binary Classification", "Multiclass", "Regression"],
        )
        task_map = {
            "Auto Detect": None,
            "Binary Classification": "binary",
            "Multiclass": "multiclass",
            "Regression": "regression",
        }
        manual_task = task_map[task_override]

        cv_folds  = st.slider("CV Folds (k)", 3, 10, 5)
        workers   = st.slider("Parallel Workers", 1, 8, 4)
        lambda1   = st.slider("λ₁ Variance Penalty", 0.0, 1.0, 0.30, step=0.05)
        lambda2   = st.slider("λ₂ Overfit Penalty",  0.0, 1.0, 0.40, step=0.05)
        enable_tune = st.checkbox("🔧 Hyperparameter Tuning (Randomized)", value=False)
        enable_shap = st.checkbox("🔬 SHAP Explainability", value=True)
        tune_iters  = st.slider("Tuning Iterations", 5, 50, 20) if enable_tune else 20

        st.divider()
        run_sup = st.button("🚀 Run Supervised Analysis", width='stretch')

    # ── Unsupervised Config ──
    elif mode == "unsupervised" and st.session_state["df"] is not None:
        st.markdown("**Unsupervised Settings**")
        k_min  = st.slider("k min", 2, 4, 2)
        k_max  = st.slider("k max", 4, 15, 8)
        contam = st.slider("Anomaly Contamination", 0.01, 0.30, 0.10, step=0.01)
        algos_all = ["KMeans", "MiniBatch KMeans", "Agglomerative", "Gaussian Mixture"]
        sel_algos = st.multiselect("Clustering Algorithms", algos_all, default=algos_all)
        inc_tsne  = st.checkbox("Include t-SNE", value=True)
        inc_umap  = st.checkbox("Include UMAP", value=True)

        st.divider()
        run_unsup = st.button("🚀 Run Unsupervised Analysis", width='stretch')

    st.divider()
    st.markdown(
        '<div style="font-size:10px; color:var(--muted); line-height:1.6;">'
        'IntelliML · scikit-learn · XGBoost<br>LightGBM · SHAP · Optuna · UMAP<br>'
        'scipy.stats · plotly</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: metric badge HTML
# ─────────────────────────────────────────────────────────────────────────────
def badge(text: str, kind: str = "cyan") -> str:
    return f'<span class="badge-{kind}">{text}</span>'


def rec_box(content: str, title: str = "🤖 RECOMMENDATION"):
    st.markdown(
        f'<div class="rec-box"><div style="font-size:11px;font-weight:700;'
        f'color:var(--accent);letter-spacing:1px;margin-bottom:10px;">{title}</div>'
        f'<p style="font-size:14px;line-height:1.7;margin:0;">{content}</p></div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN AREA: No data
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["df"] is None:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;">
      <div style="font-size:48px; margin-bottom:8px;">🧠</div>
      <div style="font-size:28px; font-weight:800; margin-bottom:8px;">
        Intelligent ML Model Selection System
      </div>
      <div style="color:var(--muted); font-size:15px; max-width:600px; margin:0 auto 32px;">
        Upload a CSV dataset or select a sample → Auto-detect task → Parallel cross-validation
        → Statistical verification → Ranked recommendation
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature grid
    fcols = st.columns(3)
    features = [
        ("⚡ Parallel CV Engine", "Multi-model cross-validation via ThreadPoolExecutor, StratifiedKFold/KFold, configurable workers"),
        ("🧬 Auto Task Detection", "Intelligently detects Binary, Multiclass, or Regression and selects optimal metrics"),
        ("📊 Statistical Comparison", "Paired t-test (scipy.stats) between top models — flags statistical equivalence"),
        ("🔧 Hyperparameter Tuning", "RandomizedSearchCV + Optuna-based Bayesian optimization for XGBoost & LightGBM"),
        ("🔬 SHAP Explainability", "TreeExplainer & KernelExplainer for feature attribution on the winning model"),
        ("🔍 Cluster Discovery", "K-Means, DBSCAN, OPTICS, Agglomerative, GMM with automatic optimal k selection"),
        ("📐 Quality Metrics", "Silhouette, Davies-Bouldin, Calinski-Harabasz, Adjusted Rand Index stability"),
        ("🚨 Anomaly Detection", "IsolationForest + LocalOutlierFactor with combined ensemble flagging"),
        ("🗺️ Dim Reduction", "PCA, t-SNE, UMAP projections with interactive Plotly scatter plots"),
    ]
    for i, (title, desc) in enumerate(features):
        with fcols[i % 3]:
            st.markdown(
                f'<div style="background:var(--card);border:1px solid var(--border);'
                f'border-radius:10px;padding:16px;margin-bottom:12px;">'
                f'<div style="font-size:14px;font-weight:700;margin-bottom:6px;">{title}</div>'
                f'<div style="font-size:12px;color:var(--muted);line-height:1.5;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  DATA PREVIEW TAB
# ─────────────────────────────────────────────────────────────────────────────
df = st.session_state["df"]

# Profile (lightweight, cached)
@st.cache_data
def _profile(df_json: str, target: str = None):
    import json
    return profile_dataset(pd.read_json(io.StringIO(df_json)), target)

df_json_key = df.to_json() if df is not None else None

# ─────────────────────────────────────────────────────────────────────────────
#  SUPERVISED ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if mode == "supervised" and "run_sup" in dir() and run_sup:
    st.session_state["ran_analysis"] = False

    # ── Profile ──
    profile = profile_dataset(df, target_col)
    if manual_task:
        profile.task_type = manual_task
    st.session_state["profile"] = profile

    # ── Task detection ──
    task = profile.task_type
    primary_metric, _ = auto_select_metric(task)

    # ── Progress bar ──
    prog_bar = st.progress(0)
    status   = st.status("⚡ Running parallel cross-validation engine...", expanded=True)

    # ── Prepare data ──
    X, y, _, label_enc = prepare_supervised_data(df, target_col, profile)
    model_count = [0]

    def on_model_done(name):
        model_count[0] += 1
        total = 9 if task in ("binary", "multiclass") else 10
        prog_bar.progress(int(model_count[0] / total * 70))
        status.write(f"  ✓ {name}")

    # ── Parallel CV ──
    with status:
        status.write(f"Task: **{task.upper()}** · Metric: **{metric_label(primary_metric)}** · Folds: **{cv_folds}**")
        results = run_parallel_cv(
            X, y, profile, cv_folds=cv_folds,
            lambda1=lambda1, lambda2=lambda2,
            max_workers=workers,
            progress_cb=on_model_done,
        )
        prog_bar.progress(70)
        status.write("🔬 Running statistical comparison...")

        # ── Statistical comparison ──
        pval, stat_equiv = statistical_comparison(results[0], results[1]) if len(results) >= 2 else (1.0, True)
        prog_bar.progress(75)

        # ── Train final model ──
        status.write("🏋️ Training final model on 80% split...")
        pipeline, X_test, y_test, y_pred, y_pred_proba = train_final_model(
            X, y, profile, results[0].name,
            tune=enable_tune, tune_iters=tune_iters,
        )
        prog_bar.progress(85)

        # ── Metrics ──
        if task in ("binary", "multiclass"):
            clf_metrics = compute_classification_metrics(y_test, y_pred, y_pred_proba, task, label_enc)
            reg_metrics = None
        else:
            reg_metrics = compute_regression_metrics(y_test, y_pred)
            clf_metrics = None

        # ── Feature importance ──
        feat_names = profile.numeric_cols + profile.categorical_cols
        fi_df = extract_feature_importance(pipeline, feat_names)
        prog_bar.progress(90)

        # ── SHAP ──
        shap_vals = shap_X_tr = None
        if enable_shap:
            status.write("🔬 Computing SHAP values...")
            shap_vals, _, shap_X_tr, _ = compute_shap(pipeline, X_test, profile)

        prog_bar.progress(100)
        status.write("✅ Analysis complete!")

    status.update(label="✅ Analysis Complete", state="complete")
    prog_bar.empty()

    # ── Store results ──
    st.session_state.update({
        "sup_results": results, "final_pipeline": pipeline,
        "X_test": X_test, "y_test": y_test,
        "y_pred": y_pred, "y_pred_proba": y_pred_proba,
        "clf_metrics": clf_metrics, "reg_metrics": reg_metrics,
        "fi_df": fi_df, "shap_vals": shap_vals, "shap_X_tr": shap_X_tr,
        "ran_analysis": True, "stat_equiv": stat_equiv, "pval": pval,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  UNSUPERVISED ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if mode == "unsupervised" and "run_unsup" in dir() and run_unsup:
    prog_bar = st.progress(0)
    status   = st.status("🔍 Running unsupervised pipeline...", expanded=True)

    def unsup_cb(msg, pct):
        status.write(f"  → {msg}")
        prog_bar.progress(pct)

    with status:
        results_u, df_out = run_unsupervised_pipeline(
            df, k_min=k_min, k_max=k_max,
            contamination=contam,
            include_tsne=inc_tsne, include_umap=inc_umap,
            algorithms=sel_algos if sel_algos else None,
            progress_cb=unsup_cb,
        )
        prog_bar.progress(100)
        status.write("✅ Pipeline complete!")

    status.update(label="✅ Unsupervised Analysis Complete", state="complete")
    prog_bar.empty()
    st.session_state.update({
        "unsup_results": results_u,
        "unsup_df_out": df_out,
        "ran_analysis": True,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  DATA OVERVIEW (always visible)
# ─────────────────────────────────────────────────────────────────────────────
if df is not None:
    with st.expander("📋 Dataset Overview", expanded=not st.session_state["ran_analysis"]):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Numeric", len(df.select_dtypes(include="number").columns))
        c4.metric("Missing %", f"{df.isnull().mean().mean()*100:.2f}%")
        st.dataframe(df.head(8), width='stretch', height=240)


# ─────────────────────────────────────────────────────────────────────────────
#  SUPERVISED RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if mode == "supervised" and st.session_state["sup_results"]:
    results    = st.session_state["sup_results"]
    profile    = st.session_state["profile"]
    task       = profile.task_type
    best       = results[0]
    pmet       = best.primary_metric
    stat_equiv = st.session_state.get("stat_equiv", False)
    pval       = st.session_state.get("pval", 1.0)

    st.markdown("---")
    # ── Top KPI row ──────────────────────────────────────────────────────────
    st.markdown(
        f"### 🏆 Best Model: `{best.name}` &nbsp;"
        + badge(task.upper(), "cyan")
        + "&nbsp;" + badge(metric_label(pmet), "purple"),
        unsafe_allow_html=True
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(metric_label(pmet), f"{best.cv_mean:.4f}", f"± {best.cv_std:.4f}")
    k2.metric("Train Score",      f"{best.train_score:.4f}")
    k3.metric("Overfit Gap",      f"{best.overfit_gap:.4f}",
              delta=None if best.overfit_gap < 0.10 else "⚠ HIGH",
              delta_color="inverse" if best.overfit_gap >= 0.10 else "normal")
    k4.metric("CV Stability ±",   f"{best.cv_std:.4f}",
              delta=None if not best.high_variance else "⚠ HIGH VAR",
              delta_color="inverse" if best.high_variance else "normal")
    k5.metric("Final Score",      f"{best.final_score:.4f}")

    # ── Statistical equivalence alert ────────────────────────────────────────
    if stat_equiv:
        st.info(
            f"🔬 **Statistical Note:** Top 2 models (`{results[0].name}` vs `{results[1].name}`) "
            f"are **statistically equivalent** (paired t-test p={pval:.4f} > 0.05). "
            "Either model can be deployed with similar expected performance."
        )

    # ── Recommendation box ───────────────────────────────────────────────────
    gap_flag  = "minimal" if best.overfit_gap < 0.05 else ("moderate" if best.overfit_gap < 0.10 else "elevated")
    var_flag  = "stable" if not best.high_variance else "slightly volatile"
    top_feat  = st.session_state["fi_df"]["feature"].iloc[0] if (
        st.session_state["fi_df"] is not None and len(st.session_state["fi_df"]) > 0
    ) else "N/A"
    rec_text = (
        f"<b>{best.name}</b> is recommended as the optimal model for this {task} task. "
        f"It achieves a cross-validated {metric_label(pmet)} of <b>{best.cv_mean:.4f} ± {best.cv_std:.4f}</b> "
        f"across {cv_folds} folds, with a {gap_flag} overfitting gap of {best.overfit_gap:.4f} "
        f"and {var_flag} fold variance. "
        f"The IntelliML ranking formula (FinalScore = {metric_label(pmet)} − "
        f"{lambda1:.2f}×Variance − {lambda2:.2f}×OverfitGap) assigns it a final score of "
        f"<b>{best.final_score:.4f}</b>, placing it {best.final_score - results[-1].final_score:.4f} "
        f"points above the weakest model. Top predictive feature: <b>{top_feat}</b>."
    )
    rec_box(rec_text)

    # ── Ranking formula display ───────────────────────────────────────────────
    st.markdown(
        f'<div class="formula-box">FinalScore = {metric_label(pmet)} '
        f'− <span style="color:var(--warn)">{lambda1:.2f} × Variance</span> '
        f'− <span style="color:var(--danger)">{lambda2:.2f} × OverfitGap</span></div>',
        unsafe_allow_html=True
    )

    # ─── TABS ────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Leaderboard", "⚔️ Overfitting", "📈 CV Distribution",
        "🌟 Features", "📉 Model Metrics",
        "🔬 SHAP",  "📦 Export",
    ])

    # ── Tab 1: Leaderboard ───────────────────────────────────────────────────
    with tabs[0]:
        c_l, c_r = st.columns([3, 2])
        with c_l:
            st.markdown("#### Model Rankings")
            rank_data = []
            for i, m in enumerate(results):
                status_flags = []
                if i == 0: status_flags.append("🏆 BEST")
                if m.overfitting: status_flags.append("⚠ OVERFIT")
                if m.high_variance: status_flags.append("📈 HIGH VAR")
                if stat_equiv and i in (0, 1): status_flags.append("≈ EQUIV")
                rank_data.append({
                    "Rank": f"#{i+1}",
                    "Model": m.name,
                    "CV Score": f"{m.cv_mean:.4f}",
                    "± Std": f"{m.cv_std:.4f}",
                    "Train": f"{m.train_score:.4f}",
                    "Overfit Gap": f"{m.overfit_gap:.4f}",
                    "Final Score": f"{m.final_score:.4f}",
                    "Time (s)": f"{m.training_time:.2f}",
                    "Flags": " ".join(status_flags),
                })
            st.dataframe(pd.DataFrame(rank_data), width='stretch', height=350)
        with c_r:
            st.plotly_chart(
                charts.leaderboard_bar(results, metric_label(pmet)),
                width='stretch',
            )

    # ── Tab 2: Overfitting ───────────────────────────────────────────────────
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.train_val_comparison(results), width='stretch')
        with c2:
            st.plotly_chart(charts.overfit_heatmap(results), width='stretch')

        st.markdown("#### Overfitting Summary Table")
        ov_data = [{
            "Model": m.name,
            "Train Score": f"{m.train_score:.4f}",
            "Val Score":   f"{m.val_score:.4f}",
            "Overfit Gap": f"{m.overfit_gap:.4f}",
            "Status": "⚠ OVERFIT" if m.overfitting else "✓ OK",
            "Variance": f"{m.cv_std:.4f}",
            "Var Status": "⚠ HIGH" if m.high_variance else "✓ STABLE",
        } for m in results]
        st.dataframe(pd.DataFrame(ov_data), width='stretch')

    # ── Tab 3: CV Distribution ───────────────────────────────────────────────
    with tabs[2]:
        st.plotly_chart(charts.cv_fold_boxplot(results), width='stretch')
        st.markdown("#### Fold-by-Fold Scores")
        fold_data = {"Model": [m.name for m in results]}
        max_folds = max((len(m.cv_scores) for m in results), default=0)
        for i in range(max_folds):
            fold_data[f"Fold {i+1}"] = [
                f"{m.cv_scores[i]:.4f}" if i < len(m.cv_scores) else "—"
                for m in results
            ]
        fold_data["Mean"] = [f"{m.cv_mean:.4f}" for m in results]
        fold_data["Std"]  = [f"{m.cv_std:.4f}"  for m in results]
        st.dataframe(pd.DataFrame(fold_data), width='stretch')

    # ── Tab 4: Features ──────────────────────────────────────────────────────
    with tabs[3]:
        fi_df = st.session_state["fi_df"]
        if fi_df is not None and len(fi_df) > 0:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.plotly_chart(charts.feature_importance_bar(fi_df), width='stretch')
            with c2:
                st.markdown("#### Feature Importance Table")
                st.dataframe(
                    fi_df.rename(columns={
                        "feature": "Feature",
                        "importance": "Raw Importance",
                        "importance_pct": "Importance (%)",
                    }),
                    width='stretch', height=400
                )
        else:
            st.info("Feature importance not available for this model type.")

    # ── Tab 5: Model Metrics ─────────────────────────────────────────────────
    with tabs[4]:
        y_test  = st.session_state["y_test"]
        y_pred  = st.session_state["y_pred"]
        y_proba = st.session_state["y_pred_proba"]

        if task in ("binary", "multiclass"):
            clf_m = st.session_state["clf_metrics"]
            if clf_m:
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        charts.confusion_matrix_plot(clf_m["confusion_matrix"]),
                        width='stretch'
                    )
                with c2:
                    cr = clf_m["classification_report"]
                    st.markdown("#### Classification Report")
                    cr_rows = []
                    for label, vals in cr.items():
                        if isinstance(vals, dict):
                            cr_rows.append({
                                "Class/Metric": label,
                                "Precision": f"{vals.get('precision', 0):.4f}",
                                "Recall":    f"{vals.get('recall', 0):.4f}",
                                "F1-Score":  f"{vals.get('f1-score', 0):.4f}",
                                "Support":   vals.get("support", "—"),
                            })
                    st.dataframe(pd.DataFrame(cr_rows), width='stretch')

                if task == "binary" and "roc" in clf_m:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(
                            charts.roc_curve_plot(clf_m["roc"], best.name),
                            width='stretch'
                        )
                    with c2:
                        if "prc" in clf_m:
                            st.plotly_chart(
                                charts.precision_recall_plot(clf_m["prc"], best.name),
                                width='stretch'
                            )

        else:  # regression
            reg_m = st.session_state["reg_metrics"]
            if reg_m:
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("R² Score", f"{reg_m['r2']:.4f}")
                r2.metric("RMSE",     f"{reg_m['rmse']:.4f}")
                r3.metric("MAE",      f"{reg_m['mae']:.4f}")
                r4.metric("MAPE",     f"{reg_m['mape']:.2f}%")
                st.plotly_chart(
                    charts.residual_plot(y_test, y_pred),
                    width='stretch'
                )

    # ── Tab 6: SHAP ─────────────────────────────────────────────────────────
    with tabs[5]:
        shap_vals  = st.session_state["shap_vals"]
        shap_X_tr  = st.session_state["shap_X_tr"]
        feat_names = profile.numeric_cols + profile.categorical_cols

        if shap_vals is not None:
            st.plotly_chart(
                charts.shap_bar_plot(shap_vals, feat_names, task),
                width='stretch'
            )
            st.markdown("#### SHAP Waterfall (Top Sample)")
            try:
                import shap, matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                exp = shap.Explanation(
                    values=shap_vals[0] if isinstance(shap_vals, np.ndarray) else shap_vals[0][0],
                    feature_names=feat_names[:shap_X_tr.shape[1]],
                )
                fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f1e35")
                shap.waterfall_plot(exp, show=False)
                plt.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close()
            except Exception:
                st.info("Waterfall plot unavailable for this model type. Bar chart above shows mean |SHAP|.")
        else:
            st.info("SHAP not computed. Enable SHAP in sidebar and re-run.")

    # ── Tab 7: Export ────────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("#### 📦 Download Artifacts")
        c1, c2, c3 = st.columns(3)

        with c1:
            pipeline = st.session_state["final_pipeline"]
            if pipeline:
                st.download_button(
                    "⬇ Download Model (.pkl)",
                    data=model_to_bytes(pipeline),
                    file_name=f"{best.name.replace(' ', '_').lower()}_model.pkl",
                    mime="application/octet-stream",
                    width='stretch',
                )

        with c2:
            st.download_button(
                "⬇ Leaderboard CSV",
                data=results_to_leaderboard_csv(results),
                file_name="model_leaderboard.csv",
                mime="text/csv",
                width='stretch',
            )

        with c3:
            st.download_button(
                "⬇ CV Metrics JSON",
                data=results_to_json(results),
                file_name="cv_metrics.json",
                mime="application/json",
                width='stretch',
            )

        # Deployment summary
        st.markdown("#### Deployment Summary")
        deploy_info = {
            "recommended_model":  best.name,
            "task_type":          task,
            "primary_metric":     metric_label(pmet),
            "cv_score":           f"{best.cv_mean:.4f} ± {best.cv_std:.4f}",
            "overfit_gap":        f"{best.overfit_gap:.4f}",
            "overfitting":        str(best.overfitting),
            "high_variance":      str(best.high_variance),
            "final_score":        f"{best.final_score:.4f}",
            "lambda1_variance_penalty": lambda1,
            "lambda2_overfit_penalty":  lambda2,
            "cv_folds":           cv_folds,
            "stat_equivalent_top2": str(stat_equiv),
            "ttest_pvalue":       f"{pval:.4f}",
            "random_state":       42,
        }
        st.json(deploy_info)


# ─────────────────────────────────────────────────────────────────────────────
#  UNSUPERVISED RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "unsupervised" and st.session_state["unsup_results"]:
    res    = st.session_state["unsup_results"]
    df_out = st.session_state["unsup_df_out"]
    best   = next((r for r in res.k_sweep if r.algorithm == res.best_algorithm
                   and r.k == res.best_k), res.k_sweep[0])

    st.markdown("---")
    # ── KPI Row ──────────────────────────────────────────────────────────────
    st.markdown(
        f"### 🔍 Optimal Configuration: `{res.best_algorithm}` · k={res.best_k} &nbsp;"
        + badge("UNSUPERVISED", "purple"),
        unsafe_allow_html=True
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Optimal Clusters", res.best_k)
    k2.metric("Silhouette Score", f"{best.silhouette:.4f}",
              delta="Good" if best.silhouette > 0.50 else "Fair",
              delta_color="normal" if best.silhouette > 0.50 else "off")
    k3.metric("Davies-Bouldin",   f"{best.davies_bouldin:.4f}",
              delta="Low ✓" if best.davies_bouldin < 1.0 else "High ⚠",
              delta_color="normal" if best.davies_bouldin < 1.0 else "inverse")
    k4.metric("IsolationForest Anomalies",
              f"{int(np.sum(res.anomaly_labels == -1))}",
              f"{np.mean(res.anomaly_labels == -1)*100:.1f}%")
    k5.metric("Algorithm",        res.best_algorithm)

    # ── Recommendation ───────────────────────────────────────────────────────
    n_noise = int(np.sum(res.best_labels == -1))
    noise_note = f" with {n_noise} noise points identified" if n_noise > 0 else ""
    rec_u = (
        f"<b>{res.best_algorithm}</b> with k=<b>{res.best_k}</b> clusters is the optimal configuration"
        f"{noise_note}. The cluster quality achieves a Silhouette Score of <b>{best.silhouette:.4f}</b> "
        f"(target >0.50) and a Davies-Bouldin Index of <b>{best.davies_bouldin:.4f}</b> (lower = better separation). "
        f"IsolationForest flagged <b>{np.mean(res.anomaly_labels==-1)*100:.1f}%</b> anomalies, "
        f"consistent with the configured contamination rate. "
        f"The ranking formula w₁·Silhouette − w₂·DB/10 assigned a final score of <b>{best.final_score:.4f}</b>."
    )
    rec_box(rec_u, "🤖 CLUSTER DISCOVERY RECOMMENDATION")

    # ── Formula ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="formula-box">FinalScore = '
        '<span style="color:var(--accent)">0.60 × Silhouette</span> '
        '− <span style="color:var(--warn)">0.40 × (DB / 10)</span> '
        '&nbsp;·&nbsp; Stability via Adjusted Rand Index across 3 runs</div>',
        unsafe_allow_html=True
    )

    # ── TABS ─────────────────────────────────────────────────────────────────
    utabs = st.tabs([
        "🗺️ Cluster Projections", "📐 K-Sweep Metrics",
        "📊 Stability", "🚨 Anomaly Detection",
        "👤 Cluster Profiles", "📦 Export",
    ])

    # ── Tab 1: Projections ───────────────────────────────────────────────────
    with utabs[0]:
        anomaly_mask = res.anomaly_labels == -1
        proj_tabs = st.tabs(
            ["PCA"] +
            (["t-SNE"] if res.tsne_2d is not None else []) +
            (["UMAP"]  if res.umap_2d is not None else [])
        )

        with proj_tabs[0]:
            evr = res.explained_variance_ratio
            ax1 = f"PC1 ({evr[0]*100:.1f}%)" if evr is not None and len(evr) > 0 else "PC1"
            ax2 = f"PC2 ({evr[1]*100:.1f}%)" if evr is not None and len(evr) > 1 else "PC2"
            fig = charts.cluster_scatter(
                res.pca_2d, res.best_labels, anomaly_mask,
                title=f"PCA 2D — {res.best_algorithm} k={res.best_k}",
                axis_labels=(ax1, ax2),
            )
            st.plotly_chart(fig, width='stretch')
            if evr is not None:
                st.plotly_chart(charts.explained_variance_plot(evr), width='stretch')

        idx_tab = 1
        if res.tsne_2d is not None:
            with proj_tabs[idx_tab]:
                fig = charts.cluster_scatter(
                    res.tsne_2d, res.best_labels, anomaly_mask,
                    title=f"t-SNE 2D — {res.best_algorithm} k={res.best_k}",
                    axis_labels=("t-SNE Dim 1", "t-SNE Dim 2"),
                )
                st.plotly_chart(fig, width='stretch')
            idx_tab += 1

        if res.umap_2d is not None:
            with proj_tabs[idx_tab]:
                fig = charts.cluster_scatter(
                    res.umap_2d, res.best_labels, anomaly_mask,
                    title=f"UMAP 2D — {res.best_algorithm} k={res.best_k}",
                    axis_labels=("UMAP Dim 1", "UMAP Dim 2"),
                )
                st.plotly_chart(fig, width='stretch')

    # ── Tab 2: K-Sweep ───────────────────────────────────────────────────────
    with utabs[1]:
        c1, c2, c3 = st.columns(3)
        with c1:
                st.plotly_chart(charts.silhouette_sweep(res.k_sweep), width='stretch')
        with c2:
            st.plotly_chart(charts.davies_bouldin_sweep(res.k_sweep), width='stretch')
        with c3:
            st.plotly_chart(charts.calinski_sweep(res.k_sweep), width='stretch')

        st.markdown("#### Full K-Sweep Rankings")
        sweep_data = [{
            "Rank": i+1,
            "Algorithm": r.algorithm,
            "k": r.k,
            "Silhouette": f"{r.silhouette:.4f}",
            "Davies-Bouldin": f"{r.davies_bouldin:.4f}",
            "Calinski-Harabasz": f"{r.calinski_harabasz:.1f}",
            "ARI Stability": f"{r.stability_ari:.4f}",
            "Final Score": f"{r.final_score:.4f}",
            "Status": "🏆 BEST" if i == 0 else ("✓" if r.silhouette > 0.5 else ""),
        } for i, r in enumerate(res.k_sweep[:20])]
        st.dataframe(pd.DataFrame(sweep_data), width='stretch')

    # ── Tab 3: Stability ─────────────────────────────────────────────────────
    with utabs[2]:
        st.plotly_chart(charts.stability_ari_bar(res.k_sweep), width='stretch')
        st.markdown("""
        **Adjusted Rand Index (ARI)** measures how similar cluster assignments are across multiple
        random runs. ARI = 1.0 indicates perfectly stable clusters. Values > 0.80 indicate high stability.
        ARI < 0.60 suggests the configuration is sensitive to initialization.
        """)

    # ── Tab 4: Anomaly Detection ─────────────────────────────────────────────
    with utabs[3]:
        iso_anomaly_pct = float(np.mean(res.anomaly_labels == -1) * 100)
        lof_anomaly_pct = float(np.mean(res.lof_labels == -1) * 100)
        combined_pct    = float(np.mean(
            (res.anomaly_labels == -1) & (res.lof_labels == -1)
        ) * 100)

        a1, a2, a3 = st.columns(3)
        a1.metric("IsolationForest", f"{int(np.sum(res.anomaly_labels==-1))} anomalies",
                  f"{iso_anomaly_pct:.1f}% of data")
        a2.metric("LocalOutlierFactor", f"{int(np.sum(res.lof_labels==-1))} anomalies",
                  f"{lof_anomaly_pct:.1f}% of data")
        a3.metric("Combined (Both Agree)", f"{int(np.sum((res.anomaly_labels==-1) & (res.lof_labels==-1)))}",
                  f"{combined_pct:.1f}% strong anomalies")

        st.plotly_chart(
            charts.anomaly_score_histogram(res.anomaly_scores, res.anomaly_labels),
            width='stretch'
        )

        st.markdown("#### Top 20 Most Anomalous Samples (by IsolationForest Score)")
        top_idx = np.argsort(res.anomaly_scores)[-20:][::-1]
        df_top_anon = df_out.iloc[top_idx].copy()
        df_top_anon.insert(0, "Anomaly_Score", res.anomaly_scores[top_idx].round(4))
        st.dataframe(df_top_anon, width='stretch')

    # ── Tab 5: Cluster Profiles ───────────────────────────────────────────────
    with utabs[4]:
        profiles = res.cluster_profiles
        st.plotly_chart(charts.cluster_profile_heatmap(profiles), width='stretch')

        st.markdown("#### Cluster Profile Table")
        st.dataframe(profiles, width='stretch')

        # Per-cluster summary cards
        mean_cols = [c for c in profiles.columns if c.endswith("_mean")]
        if mean_cols and "cluster" in profiles.columns:
            st.markdown("#### Cluster Summaries")
            palette = ["#00d4ff", "#7c3aed", "#10d98a", "#f59e0b", "#ef4444", "#ec4899"]
            for _, row in profiles.iterrows():
                cid = int(row["cluster"])
                col = palette[cid % len(palette)]
                size = int(row.get("size", 0))
                pct  = float(row.get("size_pct", 0))
                top_feat = max(
                    [(c.replace("_mean", ""), abs(float(row[c]))) for c in mean_cols],
                    key=lambda x: x[1]
                )[0]
                st.markdown(
                    f'<div style="border-left:3px solid {col};background:var(--card);'
                    f'border:1px solid var(--border);border-radius:8px;padding:14px 18px;'
                    f'margin-bottom:10px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<div style="font-size:14px;font-weight:700;">Cluster {cid}</div>'
                    f'<div style="font-size:12px;color:var(--muted);">{size} samples · {pct:.1f}%</div>'
                    f'</div>'
                    f'<div style="font-size:12px;color:var(--muted);margin-top:4px;">'
                    f'Top feature: <span style="color:{col};font-weight:700;">{top_feat}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── Tab 6: Export ────────────────────────────────────────────────────────
    with utabs[5]:
        st.markdown("#### 📦 Download Artifacts")
        c1, c2 = st.columns(2)

        with c1:
            st.download_button(
                "⬇ Labeled Dataset CSV",
                data=df_out.to_csv(index=False).encode(),
                file_name="clustered_data.csv",
                mime="text/csv",
                width='stretch',
            )

        with c2:
            summary = {
                "best_algorithm":   res.best_algorithm,
                "optimal_k":        res.best_k,
                "silhouette":       round(best.silhouette, 4),
                "davies_bouldin":   round(best.davies_bouldin, 4),
                "calinski_harabasz": round(best.calinski_harabasz, 4),
                "ari_stability":    round(best.stability_ari, 4),
                "final_score":      round(best.final_score, 4),
                "iso_anomaly_pct":  round(float(np.mean(res.anomaly_labels==-1)*100), 2),
                "lof_anomaly_pct":  round(float(np.mean(res.lof_labels==-1)*100), 2),
                "total_samples":    int(len(df)),
                "features_used":    df.shape[1],
                "tsne_computed":    res.tsne_2d is not None,
                "umap_computed":    res.umap_2d is not None,
            }
            st.download_button(
                "⬇ Analysis Summary JSON",
                data=json.dumps(summary, indent=2).encode(),
                file_name="unsupervised_summary.json",
                mime="application/json",
                width='stretch',
            )
            import json
            st.json(summary)


# ─────────────────────────────────────────────────────────────────────────────
#  IF NO ANALYSIS RUN YET (data loaded but not run)
# ─────────────────────────────────────────────────────────────────────────────
elif not st.session_state["ran_analysis"] and st.session_state["df"] is not None:
    st.markdown("---")
    st.info(
        "✅ **Dataset loaded.** Configure your settings in the sidebar and "
        "click **Run Analysis** to start the evaluation engine."
    )
