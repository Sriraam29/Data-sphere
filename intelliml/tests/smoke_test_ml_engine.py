import io
import traceback
import pandas as pd
import sys, os

# Ensure repository root is on sys.path when running tests from `tests/`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelliml.ml_engine import (
    profile_dataset, prepare_supervised_data,
    run_parallel_cv, train_final_model,
    compute_classification_metrics,
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

try:
    df = pd.read_csv(io.StringIO(SAMPLE_CLASSIFICATION))
    print('Loaded sample df', df.shape)

    profile = profile_dataset(df, target_col='loan_approved')
    print('Profile:', profile)

    X, y, preprocessor, label_enc = prepare_supervised_data(df, 'loan_approved', profile)
    print('Prepared X/y shapes:', X.shape, y.shape)

    results = run_parallel_cv(X, y, profile, cv_folds=3, max_workers=2)
    print('Completed parallel CV — models found:', len(results))
    for r in results[:3]:
        print(r.name, r.cv_mean, r.final_score)

    best = results[0]
    print('Training final model:', best.name)
    pipeline, X_test, y_test, y_pred, y_proba = train_final_model(X, y, profile, best.name, tune=False)
    print('Final model trained. Test size:', len(y_test))

    metrics = compute_classification_metrics(y_test, y_pred, y_proba, profile.task_type, label_enc)
    print('Metrics keys:', list(metrics.keys()))
    print('Smoke test completed OK')

except Exception as e:
    print('ERROR during smoke test')
    traceback.print_exc()
    raise
