import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import io
import pandas as pd
from intelliml.ml_engine import (
    profile_dataset, prepare_supervised_data, run_parallel_cv, train_final_model
)

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
"""


def test_run_parallel_cv_and_train_final():
    df = pd.read_csv(io.StringIO(SAMPLE_CLASSIFICATION))
    prof = profile_dataset(df, target_col='loan_approved')
    X, y, pre, le = prepare_supervised_data(df, 'loan_approved', prof)
    results = run_parallel_cv(X, y, prof, cv_folds=2, max_workers=1)
    assert isinstance(results, list)
    assert len(results) >= 1
    best = results[0]
    pipeline, X_test, y_test, y_pred, y_proba = train_final_model(X, y, prof, best.name, tune=False)
    assert hasattr(pipeline, 'predict')
    assert len(y_test) >= 1
