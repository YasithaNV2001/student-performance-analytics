"""The early-warning model and the Streamlit demo built on it."""
import pytest
from streamlit.testing.v1 import AppTest

from src import early_warning as ew
from src.config import ROOT

LOW_RISK = {'G1': 16, 'failures': 0, 'absences': 2, 'studytime': 3, 'higher': 'yes'}
HIGH_RISK = {'G1': 6, 'failures': 3, 'absences': 20, 'studytime': 1, 'higher': 'no',
             'goout': 5, 'famsup': 'no', 'schoolsup': 'no'}


@pytest.fixture(scope='module')
def pipe():
    return ew.load()


@pytest.fixture(scope='module')
def typical():
    X, _ = ew.load_data()
    return ew.typical_student(X)


def student_with(typical, changes):
    student = typical.copy()
    for col, value in changes.items():
        student.loc[0, col] = value
    return student


def test_features_leave_out_g2_and_g3():
    X, y = ew.load_data()
    assert 'G2' not in X.columns and 'G3' not in X.columns and 'G1' in X.columns
    assert X.shape == (395, 31)
    assert y.sum() == 130          # 130 of 395 Math students fail


def test_saved_model_matches_a_fresh_training_run(pipe):
    _, auc = ew.train()
    assert ew.holdout_auc(pipe) == pytest.approx(auc)
    assert auc > 0.85


def test_risk_is_higher_for_the_high_risk_example(pipe, typical):
    low = ew.fail_risk(pipe, student_with(typical, LOW_RISK))
    mid = ew.fail_risk(pipe, typical)
    high = ew.fail_risk(pipe, student_with(typical, HIGH_RISK))
    assert low < mid < 0.5 < high


def test_shap_values_add_up_to_the_risk(pipe, typical):
    student = student_with(typical, HIGH_RISK)
    explanation = ew.explain(pipe, student)
    assert len(explanation.values) == 31
    assert explanation.base_values + explanation.values.sum() == pytest.approx(ew.fail_risk(pipe, student))


def _risk_shown(at):
    return int(at.metric[0].value.rstrip('%'))


def test_app_shows_risk_explanation_and_note():
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
    assert not at.exception
    assert at.metric[0].label == 'Risk of failing Math'
    assert at.info[0].body == 'A tool to offer support early, not to label students.'
    typical_risk = _risk_shown(at)

    widgets = {'G1': at.slider, 'failures': at.select_slider, 'absences': at.number_input,
               'studytime': at.select_slider, 'higher': at.radio, 'goout': at.slider,
               'famsup': at.radio, 'schoolsup': at.radio}
    for key, value in HIGH_RISK.items():
        widgets[key](key=key).set_value(value)
    at.run()
    assert not at.exception
    assert _risk_shown(at) > 60 > typical_risk
