"""
IntelliML - Unsupervised Learning Engine
K-sweep clustering, stability evaluation, anomaly detection,
dimensionality reduction (PCA, t-SNE, UMAP).
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.cluster import (
    KMeans, MiniBatchKMeans, DBSCAN,
    AgglomerativeClustering, OPTICS,
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler

from .utils import ClusterResult, UnsupervisedResults, prepare_unsupervised_data

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# ─────────────────────────────────────────────
#  CLUSTERING ALGORITHMS
# ─────────────────────────────────────────────

def _fit_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    return model.fit_predict(X)


def _fit_minibatch_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    model = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    return model.fit_predict(X)


def _fit_agglomerative(X: np.ndarray, k: int) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=k)
    return model.fit_predict(X)


def _fit_gmm(X: np.ndarray, k: int) -> np.ndarray:
    model = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=5)
    return model.fit_predict(X)


CLUSTERING_ALGORITHMS = {
    "KMeans":           _fit_kmeans,
    "MiniBatch KMeans": _fit_minibatch_kmeans,
    "Agglomerative":    _fit_agglomerative,
    "Gaussian Mixture": _fit_gmm,
}


# ─────────────────────────────────────────────
#  CLUSTER QUALITY SCORING
# ─────────────────────────────────────────────

def _compute_cluster_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[float, float, float]:
    """Returns (silhouette, davies_bouldin, calinski_harabasz)."""
    n_unique = len(set(labels)) - (1 if -1 in labels else 0)
    if n_unique < 2:
        return 0.0, 9999.0, 0.0

    # Subsample for large datasets
    if len(X) > 5000:
        idx = np.random.choice(len(X), 5000, replace=False)
        X_s, y_s = X[idx], labels[idx]
        if len(set(y_s)) < 2:
            X_s, y_s = X, labels
    else:
        X_s, y_s = X, labels

    try:
        sil = silhouette_score(X_s, y_s, random_state=RANDOM_STATE)
    except Exception:
        sil = 0.0
    try:
        db = davies_bouldin_score(X_s, y_s)
    except Exception:
        db = 9999.0
    try:
        ch = calinski_harabasz_score(X_s, y_s)
    except Exception:
        ch = 0.0
    return float(sil), float(db), float(ch)


def _cluster_final_score(sil: float, db: float, w1: float = 0.6, w2: float = 0.4) -> float:
    """IntelliML unsupervised scoring: w1·Silhouette − w2·DB/10"""
    return w1 * sil - w2 * (db / 10.0)


# ─────────────────────────────────────────────
#  STABILITY EVALUATION
# ─────────────────────────────────────────────

def _stability_ari(X: np.ndarray, k: int, n_runs: int = 3) -> float:
    """Run KMeans multiple times, compute average ARI between runs."""
    runs = []
    for seed in range(n_runs):
        model = KMeans(n_clusters=k, random_state=seed, n_init=5)
        runs.append(model.fit_predict(X))

    ari_scores = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            ari_scores.append(adjusted_rand_score(runs[i], runs[j]))
    return float(np.mean(ari_scores)) if ari_scores else 0.0


# ─────────────────────────────────────────────
#  K SWEEP
# ─────────────────────────────────────────────

def run_k_sweep(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    algorithms: Optional[list[str]] = None,
    progress_cb: Optional[Callable] = None,
) -> list[ClusterResult]:
    """
    Test each algorithm × each k, compute metrics, return sorted results.
    """
    if algorithms is None:
        algorithms = list(CLUSTERING_ALGORITHMS.keys())

    k_max = min(k_max, len(X) // 3, 12)
    results: list[ClusterResult] = []

    for algo_name in algorithms:
        fit_fn = CLUSTERING_ALGORITHMS[algo_name]
        for k in range(k_min, k_max + 1):
            try:
                labels = fit_fn(X, k)
                sil, db, ch = _compute_cluster_metrics(X, labels, k)
                ari = _stability_ari(X, k) if "KMeans" in algo_name else 0.5
                score = _cluster_final_score(sil, db)

                results.append(ClusterResult(
                    algorithm=algo_name,
                    k=k,
                    labels=labels,
                    silhouette=sil,
                    davies_bouldin=db,
                    calinski_harabasz=ch,
                    stability_ari=ari,
                    final_score=score,
                    params={"k": k},
                ))
            except Exception:
                pass

            if progress_cb:
                progress_cb(algo_name, k)

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results


# ─────────────────────────────────────────────
#  DBSCAN / OPTICS (density-based)
# ─────────────────────────────────────────────

def run_dbscan(
    X: np.ndarray,
    eps_values: Optional[list[float]] = None,
    min_samples_values: Optional[list[int]] = None,
) -> Optional[ClusterResult]:
    """Auto-sweep DBSCAN parameters, return best result by silhouette."""
    if eps_values is None:
        # Auto-range based on data scale
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5).fit(X)
        dists, _ = nn.kneighbors(X)
        k_dist = np.sort(dists[:, -1])
        eps_values = [float(np.percentile(k_dist, p)) for p in [25, 50, 75, 90]]
    if min_samples_values is None:
        dim = X.shape[1]
        min_samples_values = [max(2, dim), max(4, dim * 2), max(8, dim * 3)]

    best = None
    for eps in eps_values:
        for ms in min_samples_values:
            try:
                model = DBSCAN(eps=eps, min_samples=ms)
                labels = model.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_pct  = np.sum(labels == -1) / len(labels)
                if n_clusters < 2 or noise_pct > 0.5:
                    continue
                sil, db, ch = _compute_cluster_metrics(X, labels, n_clusters)
                score = _cluster_final_score(sil, db)
                if best is None or score > best.final_score:
                    best = ClusterResult(
                        algorithm="DBSCAN",
                        k=n_clusters,
                        labels=labels,
                        silhouette=sil,
                        davies_bouldin=db,
                        calinski_harabasz=ch,
                        stability_ari=0.0,
                        final_score=score,
                        params={"eps": eps, "min_samples": ms},
                    )
            except Exception:
                pass
    return best


def run_optics(X: np.ndarray) -> Optional[ClusterResult]:
    """Run OPTICS and extract clusters."""
    try:
        model = OPTICS(min_samples=max(2, X.shape[1]), xi=0.05, min_cluster_size=0.05)
        labels = model.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            return None
        sil, db, ch = _compute_cluster_metrics(X, labels, n_clusters)
        return ClusterResult(
            algorithm="OPTICS", k=n_clusters, labels=labels,
            silhouette=sil, davies_bouldin=db, calinski_harabasz=ch,
            stability_ari=0.0, final_score=_cluster_final_score(sil, db),
        )
    except Exception:
        return None


# ─────────────────────────────────────────────
#  ANOMALY DETECTION
# ─────────────────────────────────────────────

def detect_anomalies(X: np.ndarray, contamination: float = 0.1) -> dict:
    """
    Run IsolationForest + LocalOutlierFactor.
    Returns dict with labels and scores.
    """
    # Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1)
    iso_labels = iso.fit_predict(X)         # -1 = anomaly, 1 = normal
    iso_scores = -iso.score_samples(X)      # Higher = more anomalous

    # Local Outlier Factor
    lof = LocalOutlierFactor(contamination=contamination, n_jobs=-1)
    lof_labels = lof.fit_predict(X)         # -1 = anomaly, 1 = normal
    lof_scores = -lof.negative_outlier_factor_

    # Combined: both agree = strong anomaly
    combined = (iso_labels == -1) & (lof_labels == -1)

    return {
        "iso_labels":  iso_labels,
        "iso_scores":  iso_scores,
        "lof_labels":  lof_labels,
        "lof_scores":  lof_scores,
        "combined":    combined,
        "iso_anomaly_pct": float(np.mean(iso_labels == -1) * 100),
        "lof_anomaly_pct": float(np.mean(lof_labels == -1) * 100),
        "combined_pct":    float(np.mean(combined) * 100),
    }


# ─────────────────────────────────────────────
#  DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────

def reduce_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """PCA reduction. Returns (projected_2d, explained_variance_ratio)."""
    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    proj = pca.fit_transform(X)
    return proj, pca.explained_variance_ratio_


def reduce_tsne(X: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    """t-SNE reduction to 2D."""
    perplexity = min(perplexity, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=RANDOM_STATE, n_iter=1000)
    return tsne.fit_transform(X)


def reduce_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> Optional[np.ndarray]:
    """UMAP reduction to 2D. Returns None if umap-learn not installed."""
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=min_dist, random_state=RANDOM_STATE)
        return reducer.fit_transform(X)
    except ImportError:
        return None


# ─────────────────────────────────────────────
#  CLUSTER PROFILES
# ─────────────────────────────────────────────

def compute_cluster_profiles(
    df_original: pd.DataFrame,
    labels: np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute mean feature values per cluster.
    Returns a DataFrame with cluster stats.
    """
    df = df_original.copy()
    df["_cluster_"] = labels

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "_cluster_"]

    if feature_names:
        numeric_cols = [c for c in numeric_cols if c in feature_names] or numeric_cols

    profiles = (
        df.groupby("_cluster_")[numeric_cols]
        .agg(["mean", "std", "count"])
        .round(4)
    )

    # Flatten column names
    profiles.columns = ["_".join(c).strip() for c in profiles.columns]
    profiles = profiles.reset_index().rename(columns={"_cluster_": "cluster"})
    profiles["size"] = profiles.filter(like="_count").iloc[:, 0].astype(int)
    size_col = profiles.filter(like="_count").columns[0]
    profiles["size_pct"] = (profiles["size"] / profiles["size"].sum() * 100).round(1)
    profiles = profiles[profiles["cluster"] != -1]  # exclude noise cluster

    return profiles


# ─────────────────────────────────────────────
#  FULL UNSUPERVISED PIPELINE
# ─────────────────────────────────────────────

def run_unsupervised_pipeline(
    df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 8,
    contamination: float = 0.10,
    include_tsne: bool = True,
    include_umap: bool = True,
    algorithms: Optional[list[str]] = None,
    progress_cb: Optional[Callable] = None,
) -> tuple[UnsupervisedResults, pd.DataFrame]:
    """
    Full unsupervised ML pipeline.
    Returns (UnsupervisedResults, original_df_with_labels).
    """
    if progress_cb: progress_cb("Preprocessing data", 5)

    X, feature_names = prepare_unsupervised_data(df)

    # ── PCA reduction ──
    if progress_cb: progress_cb("Running PCA projection", 15)
    pca_2d, evr = reduce_pca(X, n_components=2)

    # ── K-Sweep ──
    if progress_cb: progress_cb("Sweeping cluster configurations", 30)
    k_results = run_k_sweep(X, k_min=k_min, k_max=k_max,
                            algorithms=algorithms, progress_cb=None)

    # ── DBSCAN ──
    if progress_cb: progress_cb("Running DBSCAN density search", 50)
    dbscan_result = run_dbscan(X)
    if dbscan_result:
        k_results.append(dbscan_result)
        k_results.sort(key=lambda r: r.final_score, reverse=True)

    # ── OPTICS ──
    optics_result = run_optics(X)
    if optics_result:
        k_results.append(optics_result)
        k_results.sort(key=lambda r: r.final_score, reverse=True)

    # ── Best configuration ──
    best = k_results[0] if k_results else None
    if best is None:
        raise ValueError("No valid clustering found. Dataset may be too small.")

    best_labels = best.labels

    # ── Cluster Profiles ──
    if progress_cb: progress_cb("Computing cluster profiles", 60)
    profiles = compute_cluster_profiles(df, best_labels, feature_names)

    # ── Anomaly Detection ──
    if progress_cb: progress_cb("Running anomaly detection (IsolationForest + LOF)", 70)
    anomaly_info = detect_anomalies(X, contamination=contamination)

    # ── t-SNE ──
    tsne_2d = None
    if include_tsne:
        if progress_cb: progress_cb("Computing t-SNE projection", 80)
        try:
            tsne_2d = reduce_tsne(X)
        except Exception:
            pass

    # ── UMAP ──
    umap_2d = None
    if include_umap:
        if progress_cb: progress_cb("Computing UMAP projection", 90)
        umap_2d = reduce_umap(X)

    if progress_cb: progress_cb("Finalizing results", 98)

    # ── Annotate original dataframe ──
    df_out = df.copy()
    df_out["Cluster"]       = best_labels
    df_out["Anomaly_ISO"]   = np.where(anomaly_info["iso_labels"] == -1, "Anomaly", "Normal")
    df_out["Anomaly_LOF"]   = np.where(anomaly_info["lof_labels"] == -1, "Anomaly", "Normal")
    df_out["Anomaly_Score"] = anomaly_info["iso_scores"].round(4)

    results = UnsupervisedResults(
        best_algorithm=best.algorithm,
        best_k=best.k,
        best_labels=best_labels,
        k_sweep=k_results,
        cluster_profiles=profiles,
        anomaly_labels=anomaly_info["iso_labels"],
        anomaly_scores=anomaly_info["iso_scores"],
        lof_labels=anomaly_info["lof_labels"],
        pca_2d=pca_2d,
        tsne_2d=tsne_2d,
        umap_2d=umap_2d,
        explained_variance_ratio=evr,
    )

    return results, df_out
