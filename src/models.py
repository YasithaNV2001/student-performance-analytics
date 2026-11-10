"""Supervised models, baselines and early-warning models (notebook sections 6.1 to 6.8)."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from .config import RANDOM_STATE

# The same five folds for every model, so cross-validation scores are comparable
CV5 = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

# Features that depend on this year's grades; the early-warning models leave them out
GRADE_FEATURES = ['G1', 'G2', 'grade_trend']


def split(X, y):
    """Stratified 80/20 train/test split, shared by every model."""
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def make_rf():
    """Random Forest used for both UCI subjects."""
    return RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3,
                                  class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)


def make_lr():
    """Logistic Regression baseline on standardised features."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight='balanced'))


def make_gb():
    """Gradient Boosting model for the three-class xAPI task."""
    return GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=4,
                                      subsample=0.8, random_state=RANDOM_STATE)


def train_random_forest(X, y):
    """Fit the Random Forest on the 80% split; score it on the test set and with 5-fold CV."""
    X_tr, X_te, y_tr, y_te = split(X, y)
    model = make_rf().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    return {
        'model': model, 'X_train': X_tr, 'X_test': X_te, 'y_test': y_te,
        'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': accuracy_score(y_te, y_pred),
        'auc': roc_auc_score(y_te, y_prob),
        'cv': cross_val_score(model, X, y, cv=CV5, scoring='accuracy'),
    }


def train_xapi_gb(X, y):
    """Fit Gradient Boosting on the xAPI data; AUC is the macro one-vs-rest average."""
    X_tr, X_te, y_tr, y_te = split(X, y)
    model = make_gb().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)
    y_bin = label_binarize(y_te, classes=[0, 1, 2])
    return {
        'model': model, 'X_train': X_tr, 'X_test': X_te, 'y_test': y_te,
        'y_pred': y_pred, 'y_prob': y_prob, 'y_bin': y_bin,
        'accuracy': accuracy_score(y_te, y_pred),
        'auc': roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro'),
        'cv': cross_val_score(model, X, y, cv=CV5, scoring='accuracy'),
    }


def feature_importances(model, feature_names):
    """Tree-based feature importances as a Series indexed by feature name."""
    return pd.Series(dict(zip(feature_names, model.feature_importances_)))


def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, X_all, y_all):
    """Fit `model` and return accuracy, AUC, fail recall and 5-fold CV accuracy (in %)."""
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    cv = cross_val_score(model, X_all, y_all, cv=CV5, scoring='accuracy')
    auc = roc_auc_score(y_te, prob) if len(np.unique(prob)) > 1 else 0.5
    return {'Model': name,
            'Accuracy': accuracy_score(y_te, pred) * 100,
            'AUC': auc,
            'Fail recall': recall_score(y_te, pred, pos_label=0) * 100,
            'CV mean': cv.mean() * 100,
            'CV std': cv.std() * 100}


def evaluate_g2_rule(X_te, y_te, X_all, y_all):
    """Score the rule 'pass if G2 >= 10' the same way; AUC uses the raw G2 value."""
    pred = (X_te['G2'] >= 10).astype(int)
    # The rule needs no training, so CV just applies it to each test fold
    cv = [accuracy_score(y_all.iloc[te], (X_all['G2'].iloc[te] >= 10).astype(int))
          for _, te in CV5.split(X_all, y_all)]
    return {'Model': 'Rule: G2 >= 10',
            'Accuracy': accuracy_score(y_te, pred) * 100,
            'AUC': roc_auc_score(y_te, X_te['G2']),
            'Fail recall': recall_score(y_te, pred, pos_label=0) * 100,
            'CV mean': np.mean(cv) * 100,
            'CV std': np.std(cv) * 100}


def compare_baselines(X, y):
    """Majority class, the G2 >= 10 rule, Logistic Regression and Random Forest on one split."""
    X_tr, X_te, y_tr, y_te = split(X, y)
    rows = [
        evaluate_model('Majority class', DummyClassifier(strategy='most_frequent'),
                       X_tr, X_te, y_tr, y_te, X, y),
        evaluate_g2_rule(X_te, y_te, X, y),
        evaluate_model('Logistic Regression', make_lr(), X_tr, X_te, y_tr, y_te, X, y),
        evaluate_model('Random Forest', make_rf(), X_tr, X_te, y_tr, y_te, X, y),
    ]
    return pd.DataFrame(rows)


def early_warning_models(X, y):
    """Models without this year's grades, plus a Random Forest that keeps only G1.

    Returns the results table and (fitted no-grades Random Forest, X_test, y_test) for SHAP.
    """
    X_early = X.drop(columns=GRADE_FEATURES)
    X_tr, X_te, y_tr, y_te = split(X_early, y)
    rf_early = make_rf()
    rows = [
        evaluate_model('Majority class', DummyClassifier(strategy='most_frequent'),
                       X_tr, X_te, y_tr, y_te, X_early, y),
        evaluate_model('Logistic Regression', make_lr(), X_tr, X_te, y_tr, y_te, X_early, y),
        evaluate_model('Random Forest', rf_early, X_tr, X_te, y_tr, y_te, X_early, y),
    ]
    # Middle ground: the first-period grade is known about a third of the way into the year
    X_g1 = X.drop(columns=['G2', 'grade_trend'])
    X_tr_g, X_te_g, y_tr_g, y_te_g = split(X_g1, y)
    rows.append(evaluate_model('Random Forest (G1 only)', make_rf(),
                               X_tr_g, X_te_g, y_tr_g, y_te_g, X_g1, y))
    return pd.DataFrame(rows), (rf_early, X_te, y_te)


def print_results(df):
    """Print a results table from compare_baselines() or early_warning_models()."""
    print(f"  {'Model':<24} {'Accuracy':>9} {'AUC':>7} {'Fail recall':>12} {'5-Fold CV':>18}")
    print(f"  {'─'*74}")
    for _, r in df.iterrows():
        print(f"  {r['Model']:<24} {r['Accuracy']:>8.2f}% {r['AUC']:>7.3f} {r['Fail recall']:>11.1f}% "
              f"{r['CV mean']:>9.2f}% +/- {r['CV std']:.2f}%")
