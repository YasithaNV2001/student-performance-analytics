"""Early-warning model behind the Streamlit demo (app.py).

Predicts the risk of failing Math (final grade G3 below 10) from what a school
knows after the first term: the first-term grade G1 plus background, study
habits and social factors. G2 and G3 are left out.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import MODELS_DIR, PASS_MARK, PROCESSED_DIR, RANDOM_STATE

DATA_PATH = PROCESSED_DIR / 'student-mat-cleaned.csv'
MODEL_PATH = MODELS_DIR / 'early_warning.joblib'
LEFT_OUT = ['G2', 'G3']  # not known after the first term (G3 also defines the target)

LABELS = {
    'school': 'School', 'sex': 'Sex', 'age': 'Age', 'address': 'Home address',
    'famsize': 'Family size', 'Pstatus': "Parents live together", 'Medu': "Mother's education",
    'Fedu': "Father's education", 'Mjob': "Mother's job", 'Fjob': "Father's job",
    'reason': 'Reason for choosing the school', 'guardian': 'Guardian',
    'traveltime': 'Travel time to school', 'studytime': 'Weekly study time',
    'failures': 'Past failures', 'schoolsup': 'Extra support from school',
    'famsup': 'Study support from family', 'paid': 'Paid extra classes',
    'activities': 'Extra-curricular activities', 'nursery': 'Attended nursery school',
    'higher': 'Plans for higher education', 'internet': 'Internet at home',
    'romantic': 'In a relationship', 'famrel': 'Family relationships',
    'freetime': 'Free time after school', 'goout': 'Going out with friends',
    'Dalc': 'Weekday alcohol use', 'Walc': 'Weekend alcohol use', 'health': 'Health',
    'absences': 'Absences', 'G1': 'First-term grade (G1)',
}

# Readable versions of coded values, used in the app and the SHAP chart
VALUE_LABELS = {
    'studytime': {1: 'under 2 h', 2: '2-5 h', 3: '5-10 h', 4: 'over 10 h'},
    'traveltime': {1: 'under 15 min', 2: '15-30 min', 3: '30-60 min', 4: 'over 1 h'},
    'Medu': {0: 'none', 1: 'primary', 2: 'lower secondary', 3: 'secondary', 4: 'higher'},
    'Fedu': {0: 'none', 1: 'primary', 2: 'lower secondary', 3: 'secondary', 4: 'higher'},
    'school': {'GP': 'Gabriel Pereira', 'MS': 'Mousinho da Silveira'},
    'address': {'U': 'urban', 'R': 'rural'},
    'famsize': {'LE3': '3 or fewer', 'GT3': 'more than 3'},
    'Pstatus': {'T': 'yes', 'A': 'no'},
}
SCALE_1_TO_5 = ['famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']


def load_data(path=DATA_PATH):
    """Features (every column except G2 and G3) and the target fail = 1 when G3 < 10."""
    df = pd.read_csv(path)
    X = df.drop(columns=LEFT_OUT)
    y = (df['G3'] < PASS_MARK).astype(int).rename('fail')
    return X, y


def text_columns(X):
    """Names of the text (categorical) columns."""
    return X.select_dtypes(include=['object', 'string']).columns.tolist()


def build_pipeline(X):
    """One-hot encode the text columns, pass the numbers through, then a Random Forest."""
    prep = ColumnTransformer(
        [('text', OneHotEncoder(handle_unknown='ignore', sparse_output=False), text_columns(X))],
        remainder='passthrough')
    model = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([('prep', prep), ('model', model)])


def split(X, y):
    """Stratified 80/20 train/test split."""
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def train(path=DATA_PATH):
    """Fit the pipeline on the 80% split; returns (pipeline, AUC on the 20% test split)."""
    X, y = load_data(path)
    X_tr, X_te, y_tr, y_te = split(X, y)
    pipe = build_pipeline(X).fit(X_tr, y_tr)
    return pipe, roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])


def holdout_auc(pipe, path=DATA_PATH):
    """AUC of a fitted pipeline on the same 20% test split."""
    X, y = load_data(path)
    _, X_te, _, y_te = split(X, y)
    return roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])


def cv_auc(path=DATA_PATH):
    """5-fold cross-validated AUC: steadier than one test split of 79 students."""
    X, y = load_data(path)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(build_pipeline(X), X, y, cv=cv, scoring='roc_auc')


def save(pipe, path=MODEL_PATH):
    """Save the fitted pipeline with joblib; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return path


def load(path=MODEL_PATH):
    """Load a pipeline saved by save()."""
    return joblib.load(path)


def typical_student(X):
    """One-row table with the most common value of each text column and the median of each number."""
    text = set(text_columns(X))
    row = {col: X[col].mode()[0] if col in text else int(round(X[col].median())) for col in X.columns}
    return pd.DataFrame([row], columns=X.columns)


def fail_risk(pipe, student):
    """Predicted probability that the student fails (0 to 1)."""
    return float(pipe.predict_proba(student)[0, 1])


def display_value(col, value):
    """Readable text for a feature value, e.g. studytime 2 -> '2-5 h'."""
    if col in VALUE_LABELS:
        return VALUE_LABELS[col].get(value, str(value))
    if col in SCALE_1_TO_5:
        return f'{value}/5'
    return str(value)


def _source_columns(prep):
    """The original column behind each column the preprocessing step outputs."""
    ohe = prep.named_transformers_['text']
    source = [col for col, cats in zip(ohe.feature_names_in_, ohe.categories_) for _ in cats]
    source += [name.split('__', 1)[1] for name in prep.get_feature_names_out()[len(source):]]
    return source


def explain(pipe, student):
    """SHAP explanation of one student's fail risk, one value per original feature.

    The one-hot columns of each text feature are added back together, so the
    values still sum to (predicted risk - average risk). Positive = more risk.
    """
    prep, model = pipe.named_steps['prep'], pipe.named_steps['model']
    sv = shap.TreeExplainer(model)(prep.transform(student))
    values = pd.Series(sv.values[0, :, 1]).groupby(np.array(_source_columns(prep))).sum()
    columns = list(student.columns)
    row = student.iloc[0]
    return shap.Explanation(
        values=values.reindex(columns).to_numpy(),
        base_values=float(sv.base_values[0, 1]),
        data=np.array([display_value(c, row[c]) for c in columns], dtype=object),
        feature_names=[LABELS.get(c, c) for c in columns],
    )
