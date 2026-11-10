"""SHAP explanations for the Portuguese early-warning model (notebook section 6.9)."""
import numpy as np
import pandas as pd
import shap

FAIL_CLASS = 0  # the UCI models predict 0 = Fail, 1 = Pass


def shap_for_fail(model, X):
    """SHAP values for the Fail class: positive values push a student towards failing."""
    explanation = shap.TreeExplainer(model)(X)   # shape: (students, features, classes)
    return explanation[:, :, FAIL_CLASS]


def top_drivers(shap_fail, X, n=8):
    """Features ranked by mean |SHAP|, with the correlation between value and SHAP as direction.

    A positive direction means a higher value of the feature raises fail risk.
    """
    mean_abs = pd.Series(np.abs(shap_fail.values).mean(axis=0),
                         index=X.columns).sort_values(ascending=False)
    top = mean_abs.index[:n]
    direction = [np.corrcoef(X[f], shap_fail[:, f].values)[0, 1] for f in top]
    return pd.DataFrame({'feature': top, 'mean_abs_shap': mean_abs[top].values,
                         'direction': direction})


def flagged_students(model, X, y):
    """Find the highest-risk student overall and the highest-risk student who actually failed."""
    fail_prob = model.predict_proba(X)[:, FAIL_CLASS]
    top = int(np.argmax(fail_prob))
    failed = np.where(y.values == 0)[0]
    idx = int(failed[np.argmax(fail_prob[failed])])
    return {'fail_prob': fail_prob,
            'top_index': top,
            'top_actual': 'Fail' if y.iloc[top] == 0 else 'Pass',
            'index': idx}
