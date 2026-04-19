"""
IntelliML - Visualization Module
All Plotly chart generators for supervised and unsupervised results.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Any

# ── Palette ──────────────────────────────────
PALETTE = [
    "#00d4ff", "#7c3aed", "#10d98a", "#f59e0b",
    "#ef4444", "#ec4899", "#f97316", "#a78bfa",
    "#34d399", "#fb923c",
]
BG        = "rgba(0,0,0,0)"
PAPER_BG  = "rgba(0,0,0,0)"
FONT_CLR  = "#e2e8f0"
GRID_CLR  = "rgba(255,255,255,0.06)"
ACCENT    = "#00d4ff"
SUCCESS   = "#10d98a"
WARN      = "#f59e0b"
DANGER    = "#ef4444"
MUTED     = "#64748b"

LAYOUT_BASE = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=BG,
    font=dict(color=FONT_CLR, family="JetBrains Mono, monospace", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
)


def _axis_style(title: str = "", dtick=None) -> dict:
    d = dict(
        title=title, color=MUTED,
        showgrid=True, gridcolor=GRID_CLR,
        zeroline=False, zerolinecolor=GRID_CLR,
        linecolor="rgba(255,255,255,0.1)",
    )
    if dtick is not None:
        d["dtick"] = dtick
    return d


# ─────────────────────────────────────────────
#  SUPERVISED CHARTS
# ─────────────────────────────────────────────

def leaderboard_bar(models, metric_label: str = "CV Score") -> go.Figure:
    """Grouped bar: CV mean ± std for all models."""
    names   = [m.name for m in models]
    means   = [m.cv_mean for m in models]
    stds    = [m.cv_std  for m in models]
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(models))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=means,
        error_y=dict(type="data", array=stds, visible=True, color="rgba(255,255,255,0.4)"),
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{m:.4f}" for m in means],
        textposition="outside",
        textfont=dict(size=10),
        name=metric_label,
        hovertemplate="<b>%{x}</b><br>" + metric_label + ": %{y:.4f}<br>±%{error_y.array:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Model Leaderboard — {metric_label}", font=dict(size=13)),
        xaxis=_axis_style(),
        yaxis=_axis_style(metric_label),
        showlegend=False,
    )
    return fig


def train_val_comparison(models) -> go.Figure:
    """Side-by-side train vs validation score."""
    names = [m.name for m in models]
    train = [m.train_score for m in models]
    val   = [m.val_score   for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=train, name="Train Score",
                         marker_color=ACCENT, opacity=0.7,
                         hovertemplate="<b>%{x}</b><br>Train: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Bar(x=names, y=val, name="Validation Score",
                         marker_color=SUCCESS, opacity=0.85,
                         hovertemplate="<b>%{x}</b><br>Val: %{y:.4f}<extra></extra>"))
    fig.update_layout(
        **LAYOUT_BASE,
        barmode="group",
        title=dict(text="Train vs Validation Score", font=dict(size=13)),
        xaxis=_axis_style(),
        yaxis=_axis_style("Score"),
    )
    return fig


def overfit_heatmap(models) -> go.Figure:
    """Bubble chart: CV score vs overfit gap, bubble = std."""
    fig = go.Figure()
    for i, m in enumerate(models):
        size = max(12, m.cv_std * 500)
        color = DANGER if m.overfitting else (WARN if m.high_variance else SUCCESS)
        fig.add_trace(go.Scatter(
            x=[m.overfit_gap], y=[m.cv_mean],
            mode="markers+text",
            marker=dict(size=size, color=color, opacity=0.8,
                        line=dict(width=1, color="rgba(255,255,255,0.3)")),
            text=[m.name.split()[0]],
            textposition="top center",
            textfont=dict(size=9),
            name=m.name,
            hovertemplate=(
                f"<b>{m.name}</b><br>"
                "Overfit Gap: %{x:.4f}<br>"
                "CV Score: %{y:.4f}<br>"
                f"Std: {m.cv_std:.4f}<extra></extra>"
            ),
        ))
    # Danger zone
    fig.add_vrect(x0=0.10, x1=max(m.overfit_gap for m in models) + 0.02,
                  fillcolor="rgba(239,68,68,0.05)",
                  annotation_text="⚠ High Overfit Zone",
                  annotation_position="top left",
                  annotation_font_size=9, line_width=0)
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Overfitting Analysis — CV Score vs Gap", font=dict(size=13)),
        xaxis=_axis_style("Overfit Gap (|Train − Val|)"),
        yaxis=_axis_style("CV Score"),
        showlegend=True,
    )
    return fig


def cv_fold_boxplot(models) -> go.Figure:
    """Box plots of CV fold scores per model."""
    fig = go.Figure()
    for i, m in enumerate(models):
        if not m.cv_scores:
            continue
        fig.add_trace(go.Box(
            y=m.cv_scores,
            name=m.name,
            boxpoints="all",
            jitter=0.3,
            marker=dict(color=PALETTE[i % len(PALETTE)], size=5),
            line=dict(color=PALETTE[i % len(PALETTE)]),
            fillcolor=f"rgba({int(PALETTE[i%len(PALETTE)][1:3],16)},"
                      f"{int(PALETTE[i%len(PALETTE)][3:5],16)},"
                      f"{int(PALETTE[i%len(PALETTE)][5:7],16)},0.15)",
            hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Cross-Validation Fold Score Distribution", font=dict(size=13)),
        xaxis=_axis_style(),
        yaxis=_axis_style("Score"),
        showlegend=False,
    )
    return fig


def feature_importance_bar(df_fi: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    df = df_fi.head(top_n).sort_values("importance")
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]

    fig = go.Figure(go.Bar(
        x=df["importance_pct"], y=df["feature"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in df["importance_pct"]],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Feature Importance — Top {top_n}", font=dict(size=13)),
        xaxis=_axis_style("Relative Importance (%)"),
        yaxis=_axis_style(),
        height=max(300, top_n * 28),
    )
    return fig


def roc_curve_plot(roc_data: dict, model_name: str) -> go.Figure:
    fpr  = roc_data["fpr"]
    tpr  = roc_data["tpr"]
    auc_score = roc_data["auc"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=MUTED, dash="dash", width=1),
        name="Random Classifier (AUC=0.50)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color=ACCENT, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba(0,212,255,0.08)",
        name=f"{model_name} (AUC={auc_score:.4f})",
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="ROC Curve", font=dict(size=13)),
        xaxis=_axis_style("False Positive Rate"),
        yaxis=_axis_style("True Positive Rate"),
    )
    return fig


def precision_recall_plot(prc_data: dict, model_name: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=prc_data["recall"], y=prc_data["precision"],
        mode="lines", line=dict(color="#7c3aed", width=2.5),
        fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
        name=f"{model_name} (AP={prc_data['ap']:.4f})",
        hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Precision-Recall Curve", font=dict(size=13)),
        xaxis=_axis_style("Recall"),
        yaxis=_axis_style("Precision"),
    )
    return fig


def confusion_matrix_plot(cm: np.ndarray, labels=None) -> go.Figure:
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    text   = [[f"{cm[i,j]}<br>({cm_pct[i,j]:.1f}%)" for j in range(cm.shape[1])]
              for i in range(cm.shape[0])]

    fig = go.Figure(go.Heatmap(
        z=cm_pct, x=labels, y=labels,
        colorscale=[[0, "rgba(0,212,255,0.05)"], [0.5, "rgba(0,212,255,0.3)"], [1, ACCENT]],
        text=text, texttemplate="%{text}",
        textfont=dict(size=11, color=FONT_CLR),
        showscale=True,
        hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Confusion Matrix", font=dict(size=13)),
        xaxis=dict(title="Predicted", color=MUTED),
        yaxis=dict(title="Actual", color=MUTED, autorange="reversed"),
    )
    return fig


def residual_plot(y_true, y_pred) -> go.Figure:
    residuals = np.array(y_pred) - np.array(y_true)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Residuals vs Predicted", "Residual Distribution"])

    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(color=ACCENT, size=5, opacity=0.6),
        name="Residuals",
        hovertemplate="Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_color=WARN, line_dash="dash", row=1, col=1)

    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=30,
        marker_color=ACCENT, opacity=0.7, name="Distribution",
    ), row=1, col=2)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Residual Analysis", font=dict(size=13)),
        showlegend=False,
    )
    return fig


def shap_bar_plot(shap_values, feature_names: list[str], task: str) -> go.Figure:
    """SHAP mean absolute value bar chart."""
    try:
        if isinstance(shap_values, list):
            # multiclass: average across classes
            vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            vals = np.abs(shap_values)

        mean_abs = np.mean(vals, axis=0)
        min_len  = min(len(mean_abs), len(feature_names))
        mean_abs = mean_abs[:min_len]
        fnames   = feature_names[:min_len]

        df = pd.DataFrame({"feature": fnames, "shap": mean_abs})
        df = df.sort_values("shap").tail(15)

        fig = go.Figure(go.Bar(
            x=df["shap"], y=df["feature"],
            orientation="h",
            marker=dict(
                color=df["shap"],
                colorscale=[[0, "rgba(0,212,255,0.3)"], [1, ACCENT]],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in df["shap"]],
            textposition="outside",
            textfont=dict(size=9),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            **LAYOUT_BASE,
            title=dict(text="SHAP — Mean Absolute Feature Impact", font=dict(size=13)),
            xaxis=_axis_style("Mean |SHAP Value|"),
            yaxis=_axis_style(),
            height=max(300, len(df) * 28),
        )
        return fig
    except Exception:
        return go.Figure()


# ─────────────────────────────────────────────
#  UNSUPERVISED CHARTS
# ─────────────────────────────────────────────

def cluster_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    anomaly_mask: Optional[np.ndarray] = None,
    title: str = "Cluster Projection",
    axis_labels: tuple[str, str] = ("Dim 1", "Dim 2"),
) -> go.Figure:
    """2D scatter with cluster colors and optional anomaly overlay."""
    fig = go.Figure()
    unique_labels = sorted(set(labels) - {-1})

    for ci in unique_labels:
        mask = labels == ci
        clr  = PALETTE[int(ci) % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers",
            marker=dict(color=clr, size=6, opacity=0.75,
                        line=dict(width=0.5, color="rgba(255,255,255,0.2)")),
            name=f"Cluster {ci}",
            hovertemplate=f"Cluster {ci}<br>{axis_labels[0]}: %{{x:.3f}}<br>{axis_labels[1]}: %{{y:.3f}}<extra></extra>",
        ))

    # Noise points (DBSCAN/OPTICS label=-1)
    noise_mask = labels == -1
    if noise_mask.any():
        fig.add_trace(go.Scatter(
            x=coords[noise_mask, 0], y=coords[noise_mask, 1],
            mode="markers",
            marker=dict(color=MUTED, size=4, opacity=0.4, symbol="x"),
            name="Noise",
        ))

    # Anomalies
    if anomaly_mask is not None and anomaly_mask.any():
        fig.add_trace(go.Scatter(
            x=coords[anomaly_mask, 0], y=coords[anomaly_mask, 1],
            mode="markers",
            marker=dict(color=DANGER, size=10, opacity=0.9, symbol="star",
                        line=dict(width=1, color="white")),
            name="Anomaly",
            hovertemplate="⚠ Anomaly<br>%{x:.3f}, %{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=13)),
        xaxis=_axis_style(axis_labels[0]),
        yaxis=_axis_style(axis_labels[1]),
        height=420,
    )
    return fig


def silhouette_sweep(k_results) -> go.Figure:
    """Line + area chart of silhouette score vs k."""
    from collections import defaultdict
    by_algo = defaultdict(list)
    for r in k_results:
        if r.algorithm not in ("DBSCAN", "OPTICS"):
            by_algo[r.algorithm].append((r.k, r.silhouette))

    fig = go.Figure()
    for i, (algo, pts) in enumerate(by_algo.items()):
        pts.sort(key=lambda x: x[0])
        ks, sils = zip(*pts)
        fig.add_trace(go.Scatter(
            x=ks, y=sils, mode="lines+markers",
            name=algo, line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor=f"rgba({int(PALETTE[i%len(PALETTE)][1:3],16)},"
                      f"{int(PALETTE[i%len(PALETTE)][3:5],16)},"
                      f"{int(PALETTE[i%len(PALETTE)][5:7],16)},0.05)",
            hovertemplate=f"{algo}<br>k=%{{x}}<br>Silhouette=%{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Silhouette Score vs Number of Clusters", font=dict(size=13)),
        xaxis=_axis_style("Number of Clusters (k)", dtick=1),
        yaxis=_axis_style("Silhouette Score"),
    )
    return fig


def davies_bouldin_sweep(k_results) -> go.Figure:
    """Davies-Bouldin sweep chart (lower = better)."""
    from collections import defaultdict
    by_algo = defaultdict(list)
    for r in k_results:
        if r.algorithm not in ("DBSCAN", "OPTICS") and r.davies_bouldin < 9000:
            by_algo[r.algorithm].append((r.k, r.davies_bouldin))

    fig = go.Figure()
    for i, (algo, pts) in enumerate(by_algo.items()):
        pts.sort(key=lambda x: x[0])
        ks, dbs = zip(*pts)
        fig.add_trace(go.Scatter(
            x=ks, y=dbs, mode="lines+markers",
            name=algo, line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            marker=dict(size=7),
            hovertemplate=f"{algo}<br>k=%{{x}}<br>DB Index=%{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Davies-Bouldin Index vs k  (↓ better)", font=dict(size=13)),
        xaxis=_axis_style("Number of Clusters (k)", dtick=1),
        yaxis=_axis_style("Davies-Bouldin Index"),
    )
    return fig


def calinski_sweep(k_results) -> go.Figure:
    """Calinski-Harabasz sweep."""
    from collections import defaultdict
    by_algo = defaultdict(list)
    for r in k_results:
        if r.algorithm not in ("DBSCAN", "OPTICS"):
            by_algo[r.algorithm].append((r.k, r.calinski_harabasz))

    fig = go.Figure()
    for i, (algo, pts) in enumerate(by_algo.items()):
        pts.sort(key=lambda x: x[0])
        ks, chs = zip(*pts)
        fig.add_trace(go.Scatter(
            x=ks, y=chs, mode="lines+markers",
            name=algo, line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            marker=dict(size=7),
            hovertemplate=f"{algo}<br>k=%{{x}}<br>CH=%{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Calinski-Harabasz Score vs k  (↑ better)", font=dict(size=13)),
        xaxis=_axis_style("Number of Clusters (k)", dtick=1),
        yaxis=_axis_style("Calinski-Harabasz Score"),
    )
    return fig


def anomaly_score_histogram(scores: np.ndarray, iso_labels: np.ndarray) -> go.Figure:
    """Histogram of anomaly scores colored by decision."""
    is_ano = iso_labels == -1
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores[~is_ano], name="Normal",
        marker_color=SUCCESS, opacity=0.7, nbinsx=40,
        hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=scores[is_ano], name="Anomaly",
        marker_color=DANGER, opacity=0.8, nbinsx=20,
        hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        barmode="overlay",
        title=dict(text="Anomaly Score Distribution (IsolationForest)", font=dict(size=13)),
        xaxis=_axis_style("Anomaly Score (higher = more anomalous)"),
        yaxis=_axis_style("Count"),
    )
    return fig


def cluster_profile_heatmap(profiles: pd.DataFrame) -> go.Figure:
    """Heatmap of cluster mean feature values (normalized)."""
    mean_cols = [c for c in profiles.columns if c.endswith("_mean")]
    if not mean_cols or "cluster" not in profiles.columns:
        return go.Figure()

    z = profiles[mean_cols].values
    feat_names = [c.replace("_mean", "") for c in mean_cols]
    cluster_ids = [f"Cluster {c}" for c in profiles["cluster"]]

    # Normalize column-wise for visual comparison
    z_norm = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-9)

    fig = go.Figure(go.Heatmap(
        z=z_norm, x=feat_names, y=cluster_ids,
        colorscale="RdBu",
        text=np.round(z, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="Cluster: %{y}<br>Feature: %{x}<br>Mean: %{text}<extra></extra>",
        colorbar=dict(title="Z-Score", tickfont=dict(size=9)),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Cluster Feature Profiles (Z-Scored Means)", font=dict(size=13)),
        xaxis=dict(color=MUTED, tickangle=-30),
        yaxis=dict(color=MUTED),
        height=max(300, len(cluster_ids) * 60 + 120),
    )
    return fig


def stability_ari_bar(k_results) -> go.Figure:
    """Bar chart of ARI stability score per k."""
    pts = [(r.k, r.stability_ari) for r in k_results
           if r.algorithm == "KMeans" and r.stability_ari > 0]
    if not pts:
        return go.Figure()
    pts.sort(key=lambda x: x[0])
    ks, aris = zip(*pts)

    fig = go.Figure(go.Bar(
        x=[f"k={k}" for k in ks], y=aris,
        marker=dict(
            color=aris,
            colorscale=[[0, "rgba(239,68,68,0.6)"], [0.5, WARN], [1, SUCCESS]],
            showscale=False,
        ),
        text=[f"{a:.3f}" for a in aris],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="k=%{x}<br>ARI Stability: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Cluster Stability — Adjusted Rand Index (↑ = more stable)", font=dict(size=13)),
        xaxis=_axis_style("Number of Clusters"),
        yaxis=_axis_style("ARI (0–1)"),
        shapes=[dict(type="line", x0=-0.5, x1=len(ks)-0.5,
                     y0=0.80, y1=0.80,
                     line=dict(color=SUCCESS, dash="dash", width=1))],
        annotations=[dict(x=len(ks)-1, y=0.82, text="Stability threshold 0.80",
                          showarrow=False, font=dict(size=9, color=SUCCESS))],
    )
    return fig


def explained_variance_plot(evr: np.ndarray) -> go.Figure:
    """PCA explained variance ratio bar chart."""
    cumulative = np.cumsum(evr) * 100
    pct = evr * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(len(pct))],
        y=pct,
        name="Explained Variance",
        marker_color=ACCENT, opacity=0.8,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(len(cumulative))],
        y=cumulative,
        mode="lines+markers",
        name="Cumulative (%)",
        line=dict(color=SUCCESS, width=2),
        marker=dict(size=6),
    ), secondary_y=True)
    fig.add_hline(y=95, line_color=WARN, line_dash="dash",
                  secondary_y=True, annotation_text="95% threshold", annotation_font_size=9)
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="PCA Explained Variance", font=dict(size=13)),
    )
    fig.update_yaxes(title_text="Variance (%)", color=MUTED, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (%)", color=MUTED, secondary_y=True)
    return fig
