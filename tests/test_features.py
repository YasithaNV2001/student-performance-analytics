import pandas as pd
import pytest

from src import data, features


@pytest.fixture(scope='module')
def df_mat():
    return data.load_processed()[0]


def test_add_features_creates_grade_trend_and_study_absence_ratio():
    df = pd.DataFrame({'G1': [10, 12], 'G2': [13, 9], 'studytime': [2, 4], 'absences': [0, 3],
                       'Dalc': [1, 2], 'Walc': [3, 4], 'Medu': [4, 1], 'Fedu': [2, 1]})
    out = features.add_features(df)
    assert out['grade_trend'].tolist() == [3, -3]
    assert out['study_absence_ratio'].tolist() == [2.0, 1.0]   # studytime / (absences + 1)
    assert out['avg_alcohol'].tolist() == [2.0, 3.0]
    assert out['parent_edu_avg'].tolist() == [3.0, 1.0]
    assert 'grade_trend' not in df.columns                      # input is not modified


def test_preprocess_uci_adds_target_and_encodings(df_mat):
    out = features.preprocess_uci(df_mat)
    assert out.shape == (395, 54)
    assert out['pass_fail'].tolist() == (df_mat['G3'] >= 10).astype(int).tolist()
    assert set(out['higher_enc']) == {0, 1}


def test_build_features_excludes_final_grade(df_mat):
    X, y, cols = features.build_features(features.preprocess_uci(df_mat))
    assert len(cols) == 31
    assert 'G3' not in cols and 'pass_fail' not in cols
    assert {'G1', 'G2', 'grade_trend', 'study_absence_ratio'} <= set(cols)
    assert len(X) == len(y) == 395


def test_preprocess_xapi_engagement_score():
    df_x = features.preprocess_xapi(data.load_processed()[2])
    expected = df_x[features.BEHAV_COLS].mean(axis=1)
    assert (df_x['engagement_score'] == expected).all()
    assert set(df_x['Class_enc']) == {0, 1, 2}
