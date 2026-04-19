import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import io
import pandas as pd
import pickle
from intelliml.ml_engine import (
    profile_dataset, build_preprocessor, model_to_bytes,
    results_to_leaderboard_csv, results_to_json, metric_label,
)

SAMPLE = """a,b,c,target
1,2,x,yes
2,3,y,no
3,4,x,yes
"""

def test_profile_dataset():
    df = pd.read_csv(io.StringIO(SAMPLE))
    prof = profile_dataset(df, target_col='target')
    assert prof.n_rows == 3
    assert 'a' in prof.numeric_cols
    assert prof.target_col == 'target'


def test_build_preprocessor_and_model_bytes():
    df = pd.read_csv(io.StringIO(SAMPLE))
    prof = profile_dataset(df, target_col='target')
    pre = build_preprocessor(prof.numeric_cols, prof.categorical_cols)
    # preprocessor should be fit-transformable
    X = df.drop(columns=['target'])
    pre.fit(X)

    # model_to_bytes should serialize an object
    b = model_to_bytes({'foo': 'bar'})
    assert isinstance(b, (bytes, bytearray))
    obj = pickle.loads(b)
    assert obj['foo'] == 'bar'


def test_results_exports_and_metric_label():
    # create fake ModelResult-like objects
    class M:
        def __init__(self, name):
            self.name = name
            self.cv_mean = 0.9
            self.cv_std = 0.01
            self.train_score = 0.92
            self.val_score = 0.89
            self.overfit_gap = 0.03
            self.training_time = 0.1
            self.final_score = 0.85
            self.cv_scores = [0.9, 0.88, 0.92]
            self.high_variance = False
            self.overfitting = False
            self.primary_metric = 'accuracy'

    models = [M('m1'), M('m2')]
    csvb = results_to_leaderboard_csv(models)
    assert b'model' in csvb
    jsonb = results_to_json(models)
    assert b'model' in jsonb
    assert metric_label('roc_auc') == 'ROC-AUC'
