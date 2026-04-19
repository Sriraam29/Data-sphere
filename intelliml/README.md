# 🧠 IntelliML — Intelligent Model Selection System

A production-grade ML analysis platform with a **Streamlit** frontend and a full **scikit-learn / XGBoost / LightGBM / SHAP / Optuna** backend.

## Architecture

```
intelliml/
├── app.py                      # Streamlit UI — all pages, tabs, visualizations
├── requirements.txt
├── .streamlit/
│   └── config.toml             # Dark theme configuration
└── ml_engine/
    ├── __init__.py
    ├── utils.py                 # DataProfile, task detection, preprocessing pipelines, export
    ├── supervised.py            # Parallel CV, overfitting detection, stat tests, SHAP, Optuna
    ├── unsupervised.py          # K-Means, DBSCAN, OPTICS, GMM, IsolationForest, LOF, UMAP
    └── charts.py                # All Plotly chart generators
```

## Tech Stack

| Layer | Library |
|-------|---------|
| Frontend | **Streamlit** + custom CSS |
| Core ML | **scikit-learn** (backbone) |
| Boosting | **XGBoost**, **LightGBM** |
| Cross-Validation | `sklearn.model_selection` — StratifiedKFold / KFold |
| Statistical Tests | `scipy.stats` — paired t-test |
| Hyperparameter Tuning | `sklearn.RandomizedSearchCV` + **Optuna** |
| Explainability | **SHAP** — TreeExplainer, KernelExplainer |
| Clustering | KMeans, MiniBatchKMeans, DBSCAN, OPTICS, AgglomerativeClustering, GaussianMixture |
| Anomaly Detection | **IsolationForest**, **LocalOutlierFactor** |
| Dimensionality Reduction | PCA, t-SNE, **UMAP** |
| Cluster Stability | Adjusted Rand Index |
| Visualization | **Plotly** |

## Setup

```bash
# 1. Clone / navigate to project
cd intelliml

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

## Features

### ⚡ Supervised Module
- **Auto task detection** — Binary / Multiclass / Regression
- **9–10 models per task** evaluated in parallel via `ThreadPoolExecutor`
- **StratifiedKFold** (classification) / **KFold** (regression), configurable k
- **Ranking formula**: `FinalScore = CVScore − λ₁·Variance − λ₂·OverfitGap`
- **Paired t-test** (scipy.stats) for statistical equivalence between top 2 models
- **Overfitting analysis**: train vs val gap, high-variance flagging
- **RandomizedSearchCV** + **Optuna** Bayesian tuning for XGBoost/LightGBM
- **SHAP** TreeExplainer / KernelExplainer for model interpretation
- **ROC-AUC**, **Confusion Matrix**, **PR Curve**, **Residual plots**
- **Export**: model `.pkl`, leaderboard `.csv`, metrics `.json`

### 🔍 Unsupervised Module
- **K-sweep**: Tests KMeans, MiniBatch KMeans, Agglomerative, GMM across k=2–12
- **DBSCAN auto-sweep**: Automatic ε selection via k-distance graph
- **OPTICS**: Density-based with automatic cluster extraction
- **Scoring**: Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI Stability
- **Stability**: 3-run Adjusted Rand Index per k
- **Anomaly Detection**: IsolationForest + LocalOutlierFactor with combined ensemble
- **Projections**: PCA, t-SNE, UMAP with interactive scatter plots
- **Cluster Profiles**: Feature mean heatmap per cluster
- **Export**: labeled CSV with cluster + anomaly columns

## Ranking Formulas

**Supervised:**
```
FinalScore = PrimaryMetric − λ₁ × CVStd − λ₂ × |TrainScore − ValScore|
```
λ₁ (variance penalty) and λ₂ (overfit penalty) are configurable via sidebar.

**Unsupervised:**
```
FinalScore = 0.60 × Silhouette − 0.40 × (DaviesBouldin / 10)
```

## Sample Datasets

Three built-in sample datasets:
- **Binary Classification** — Loan approval (35 rows, 6 features)
- **Regression** — House price prediction (28 rows, 7 features)
- **Clustering** — Customer segmentation (28 rows, 7 features)
