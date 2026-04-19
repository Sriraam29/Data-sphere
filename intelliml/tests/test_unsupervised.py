import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from intelliml.ml_engine import (
    prepare_unsupervised_data, reduce_pca, run_k_sweep, compute_cluster_profiles,
    run_unsupervised_pipeline
)


def test_prepare_and_reduce():
    # create small numeric dataframe
    rng = np.random.RandomState(42)
    df = pd.DataFrame(rng.randn(30, 4), columns=['a','b','c','d'])
    X, feat = prepare_unsupervised_data(df)
    assert X.shape[0] == 30
    proj, evr = reduce_pca(X, n_components=2)
    assert proj.shape[1] == 2


def test_k_sweep_and_profiles():
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(50, 3), columns=['x','y','z'])
    X, feat = prepare_unsupervised_data(df)
    results = run_k_sweep(X, k_min=2, k_max=3, algorithms=['KMeans'])
    assert isinstance(results, list)
    if results:
        labels = results[0].labels
        profiles = compute_cluster_profiles(df, labels, feat)
        assert 'cluster' in profiles.columns


def test_full_pipeline():
    rng = np.random.RandomState(1)
    df = pd.DataFrame(rng.randn(40, 3), columns=['f1','f2','f3'])
    res, df_out = run_unsupervised_pipeline(df, k_min=2, k_max=3, include_tsne=False, include_umap=False)
    assert hasattr(res, 'best_algorithm')
    assert 'Cluster' in df_out.columns
