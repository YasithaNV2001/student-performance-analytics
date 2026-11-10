"""Feature engineering for the UCI and xAPI datasets (notebook sections 3 and 4)."""
from sklearn.preprocessing import LabelEncoder

from .config import PASS_MARK

UCI_CATEGORICAL = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                   'schoolsup', 'famsup', 'paid', 'activities', 'higher',
                   'internet', 'romantic', 'guardian', 'nursery']

# Numeric columns used by the UCI models, after the encoded categoricals
UCI_NUMERIC = ['age', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime',
               'goout', 'health', 'absences', 'G1', 'G2',
               'avg_alcohol', 'parent_edu_avg', 'grade_trend', 'study_absence_ratio']

XAPI_CATEGORICAL = ['gender', 'Semester', 'Relation', 'ParentAnsweringSurvey',
                    'ParentschoolSatisfaction', 'StudentAbsenceDays']
BEHAV_COLS = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
XAPI_CLASS_CODES = {'L': 0, 'M': 1, 'H': 2}


def label_encode(df, columns):
    """Add an integer-coded `<column>_enc` copy of each text column."""
    df = df.copy()
    encoder = LabelEncoder()
    for col in columns:
        if col in df.columns:
            df[col + '_enc'] = encoder.fit_transform(df[col].astype(str))
    return df


def add_features(df):
    """Add avg_alcohol, parent_edu_avg, grade_trend (G2 - G1) and study_absence_ratio."""
    df = df.copy()
    df['avg_alcohol'] = (df['Dalc'] + df['Walc']) / 2
    df['parent_edu_avg'] = (df['Medu'] + df['Fedu']) / 2
    df['grade_trend'] = df['G2'] - df['G1']
    df['study_absence_ratio'] = df['studytime'] / (df['absences'] + 1)  # +1 avoids division by zero
    return df


def preprocess_uci(df):
    """Pass/fail target (G3 >= 10), encoded categoricals and engineered features for a UCI dataset."""
    df = df.copy()
    df['pass_fail'] = (df['G3'] >= PASS_MARK).astype(int)
    df = label_encode(df, UCI_CATEGORICAL)
    return add_features(df)


def add_dual_subject_features(df_both):
    """Pass flags per subject, pass_both, and grade_gap (Math minus Portuguese final grade)."""
    df = df_both.copy()
    df['pass_math'] = (df['G3_math'] >= PASS_MARK).astype(int)
    df['pass_port'] = (df['G3_port'] >= PASS_MARK).astype(int)
    df['pass_both'] = ((df['pass_math'] == 1) & (df['pass_port'] == 1)).astype(int)
    df['grade_gap'] = df['G3_math'] - df['G3_port']
    return df


def preprocess_xapi(df):
    """Encode the xAPI categoricals and class (L=0, M=1, H=2); add engagement_score and two flags."""
    df = label_encode(df, XAPI_CATEGORICAL)
    df['Class_enc'] = df['Class'].map(XAPI_CLASS_CODES)
    df['engagement_score'] = df[BEHAV_COLS].mean(axis=1)
    df['high_absence'] = (df['StudentAbsenceDays'] == 'Above-7').astype(int)
    df['parent_involved'] = (df['ParentAnsweringSurvey'] == 'Yes').astype(int)
    return df


def build_features(df, target='pass_fail'):
    """UCI feature matrix (encoded categoricals, then numeric columns); returns (X, y, feature names)."""
    enc_cols = [c for c in df.columns if c.endswith('_enc')]
    feat_cols = enc_cols + [c for c in UCI_NUMERIC if c in df.columns]
    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df[target]
    return X, y, feat_cols


def xapi_features(df_x):
    """xAPI feature matrix and 0/1/2 class target; returns (X, y, feature names)."""
    feats = ([c + '_enc' for c in XAPI_CATEGORICAL if c + '_enc' in df_x.columns]
             + BEHAV_COLS + ['engagement_score'])
    feats = [f for f in feats if f in df_x.columns]
    return df_x[feats], df_x['Class_enc'], feats
