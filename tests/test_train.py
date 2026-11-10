"""Run train.py's pipeline once and check the results against the notebook's reference numbers."""
import json

import pytest

import train

# Reference numbers printed by analysis.ipynb (see the README and report.pdf)
REFERENCE_MODELS = {
    'math_random_forest':       {'accuracy': 0.8608, 'auc': 0.9419, 'cv_mean': 0.9089},
    'portuguese_random_forest': {'accuracy': 0.8769, 'auc': 0.9364, 'cv_mean': 0.9245},
    'xapi_gradient_boosting':   {'accuracy': 0.7604, 'auc': 0.9202, 'cv_mean': 0.7385},
}
REFERENCE_EARLY_WARNING_AUC = {
    'math':       {'Random Forest': 0.627, 'Random Forest (G1 only)': 0.869},
    'portuguese': {'Random Forest': 0.662, 'Random Forest (G1 only)': 0.831},
}
REFERENCE_G2_RULE_ACCURACY = {'math': 0.8734, 'portuguese': 0.9077}
TOL = 0.005


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp('pipeline')
    metrics = train.run_pipeline(processed_dir=out / 'processed', figures_dir=out / 'figures',
                                 verbose=False)
    return metrics, out


@pytest.mark.parametrize('model', REFERENCE_MODELS)
def test_model_metrics_match_reference(run, model):
    metrics, _ = run
    for name, expected in REFERENCE_MODELS[model].items():
        assert metrics['models'][model][name] == pytest.approx(expected, abs=TOL), name


def test_baselines_and_early_warning_match_reference(run):
    metrics, _ = run
    for subj, accuracy in REFERENCE_G2_RULE_ACCURACY.items():
        assert metrics['baselines'][subj]['Rule: G2 >= 10']['accuracy'] == pytest.approx(accuracy, abs=TOL)
    for subj, aucs in REFERENCE_EARLY_WARNING_AUC.items():
        for model, auc in aucs.items():
            assert metrics['early_warning'][subj][model]['auc'] == pytest.approx(auc, abs=TOL)


def test_clustering_rules_and_statistics_match_reference(run):
    metrics, _ = run
    assert metrics['clustering']['uci']['silhouette'] == pytest.approx(0.1898, abs=TOL)
    assert metrics['clustering']['xapi']['silhouette'] == pytest.approx(0.3598, abs=TOL)
    assert metrics['association_rules']['xapi']['outcome_rules'] == 11
    assert metrics['association_rules']['uci']['best_rule'] == 'alcohol_high + failures_any -> fail'
    assert metrics['statistics']['paired_ttest_math_vs_portuguese']['t'] == pytest.approx(-9.977, abs=0.01)
    assert list(metrics['shap_top_drivers_portuguese'])[:2] == ['failures', 'school_enc']


def test_pipeline_saves_all_figures_and_valid_json(run):
    metrics, out = run
    assert len(list((out / 'figures').glob('fig*.png'))) == 16
    assert len(list((out / 'processed').glob('*.csv'))) == 4
    assert json.loads(json.dumps(metrics)) == metrics
