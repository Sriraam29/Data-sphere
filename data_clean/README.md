# ⚗️ DataForge AI — Intelligent Data Cleaning & ML-Aware Preprocessing Engine

> *Upload chaos. Receive intelligence.*

A production-grade Streamlit application that takes any messy structured dataset and transforms it into a fully clean, ML-ready, statistically validated dataset — automatically.

---

## 🎯 What It Does

| Stage | What Happens |
|-------|-------------|
| **1. Schema Inference** | Detects true types (numeric, ordinal, boolean, datetime, text, ID-like) using conversion rates, cardinality, entropy, and name heuristics |
| **2. Bias Detection** | Flags fairness issues across sensitive attributes (gender, race, age, etc.) using ANOVA / Chi-square tests |
| **3. Missing Value Intelligence** | Skewness-aware imputation (mean/median/mode/forward-fill/predictive), missingness signal preservation |
| **4. Outlier Handling** | Shapiro-Wilk normality testing → Z-score (normal), IQR (non-normal), Winsorization (heavy-tailed) |
| **5. Leakage Detection** | Flags corr > 0.98 to target, post-event timestamps, heuristic name patterns |
| **6. Feature Encoding** | OneHot (low card), Frequency (medium), K-Fold Target Encoding (high card + target), Ordinal mapping |
| **7. Feature Scaling** | StandardScaler (normal), RobustScaler (skewed), MinMaxScaler (bounded) |
| **8. Feature Selection** | Variance threshold, high-correlation filter (>0.95), mutual information ranking |
| **9. ML Baseline** | LightGBM / RandomForest baseline + SHAP values + cross-validated score |

**Bonus:** Data drift detection (KS test / Chi-square) when a second dataset is uploaded.

---

## 🚀 Quick Start

### 1. Clone / download

```bash
git clone <repo> dataforge-ai
cd dataforge-ai
```

### 2. Create virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### 5. Try the example dataset

Upload `example_data/messy_dataset.csv` — a 1,030-row intentionally messy churn dataset with:
- Mixed-case categorical values
- Multiple missing value patterns (7–72% missing)
- Numeric outliers (10× income spikes)
- ID-like columns
- Near-constant columns
- Duplicate rows
- Bad date strings
- Target column: `churn`

---

## 📁 Project Structure

```
data_cleaner/
├── app.py                          # Main Streamlit UI
├── requirements.txt
├── README.md
├── generate_example_data.py        # Script to regenerate example data
│
├── modules/                        # Core cleaning engine
│   ├── __init__.py
│   ├── schema_inference.py         # Deep type detection
│   ├── missing_value_handler.py    # Intelligent imputation
│   ├── outlier_handler.py          # Distribution-aware outlier capping
│   ├── encoding_strategy.py        # Cardinality-aware encoding
│   ├── feature_scaling.py          # Distribution-aware scaling
│   ├── feature_selection.py        # Variance / corr / MI selection
│   ├── leakage_detection.py        # Data leakage scanning
│   ├── bias_detection.py           # Fairness / bias analysis
│   ├── drift_detection.py          # Statistical drift detection
│   ├── ml_task_detector.py         # Baseline model + SHAP
│   ├── memory_optimizer.py         # Downcast & category conversion
│   └── validator.py                # Post-cleaning quality checks
│
├── utils/
│   ├── __init__.py
│   └── logger.py                   # Structured transformation logger
│
└── example_data/
    └── messy_dataset.csv           # Example dirty dataset (churn)
```

---

## 📤 Exports

After cleaning, download:

| File | Description |
|------|-------------|
| `clean_dataset.csv` | Clean, ML-ready dataset |
| `clean_dataset.parquet` | Parquet version (optimised for loading) |
| `cleaning_log.json` | Full transformation log (every decision) |
| `cleaning_report.html` | Human-readable HTML report |
| `preprocessing_pipeline.joblib` | Serialised encoders + scalers |
| `schema.json` | Inferred schema for all columns |

---

## 🧠 Key Design Decisions

### Never blindly impute
- Missing value strategy is driven by skewness test, correlation strength, and missingness-target correlation
- If missingness correlates with target (>0.15), a `__was_missing` binary indicator is added before imputing

### Never blindly remove outliers
- Normality is tested per-column (Shapiro-Wilk when n≤5000, skewness/kurtosis otherwise)
- Z-score capping for normal, IQR for non-normal, Winsorization for heavy-tailed (|kurtosis| > 7)

### Target encoding is leakage-safe
- K-fold target encoding ensures validation fold statistics are never computed on training data

### Everything is logged
- Every transformation decision is stored with timestamp, stage, column, before/after values, and severity level

---

## ⚡ Performance Notes

- `@st.cache_data` on all file loading operations
- Vectorised operations throughout
- Iterative imputation skipped for datasets > 50,000 rows (falls back to median/mean)
- SHAP limited to 20,000 row sample
- ML baseline limited to 100,000 row sample
- Memory downcast typically reduces footprint 30–60%

---

## 🔧 Configuration

No configuration required — the pipeline is entirely data-driven.

Optional settings (sidebar):
- **Target column**: auto-detected by name heuristics, or specify manually
- **Run ML baseline**: toggle LightGBM/RF + SHAP computation
- **Drift dataset**: upload a second CSV to enable drift detection

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Imputation, scaling, selection, RF baseline |
| `scipy` | Statistical tests (Shapiro, KS, chi-square, ANOVA) |
| `lightgbm` | Fast gradient boosting baseline |
| `shap` | Model explainability |
| `matplotlib`, `seaborn` | Visualisations |
| `joblib` | Pipeline serialisation |
| `pyarrow` | Parquet I/O |

---

## 🐛 Troubleshooting

**`lightgbm` not available** — RandomForest is used automatically as fallback.  
**`pyarrow` not installed** — Parquet export button shows an info message.  
**Very large files (>500MB)** — Increase Streamlit's file upload limit:
```bash
streamlit run app.py --server.maxUploadSize 1024
```

---

*Built with precision. No toy logic. No hardcoded assumptions. Everything data-driven.*
