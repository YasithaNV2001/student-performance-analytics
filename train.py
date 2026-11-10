"""Run the whole analysis from the command line.

    python train.py                # clean the data, train every model, save figures and metrics
    python train.py --no-figures   # same, without drawing the 16 figures

Figures go to figures/ and the key metrics to reports/metrics.json.
"""
import argparse
import json
import platform
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # write figures to files; never open a window
import matplotlib.pyplot as plt  # noqa: E402
import sklearn  # noqa: E402

from src import clustering, data, explain, features, models, plots, rules, stats_tests  # noqa: E402
from src.config import FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE, RAW_DIR, REPORTS_DIR  # noqa: E402

# seaborn 0.13 still passes a Matplotlib argument that is deprecated; nothing to fix on our side
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning, module='seaborn')


def _r(x):
    """Round a metric to 4 decimals."""
    return round(float(x), 4)


def _sig(x):
    """Round to 4 significant figures (for p-values, which can be tiny)."""
    return float(f'{x:.4g}')


def _model_summary(res):
    return {'accuracy': _r(res['accuracy']), 'auc': _r(res['auc']),
            'cv_mean': _r(res['cv'].mean()), 'cv_std': _r(res['cv'].std())}


def _results_table(df):
    """Turn a baseline / early-warning table (in %) into {model: metrics as fractions}."""
    return {row['Model']: {'accuracy': _r(row['Accuracy'] / 100), 'auc': _r(row['AUC']),
                           'fail_recall': _r(row['Fail recall'] / 100),
                           'cv_mean': _r(row['CV mean'] / 100), 'cv_std': _r(row['CV std'] / 100)}
            for _, row in df.iterrows()}


def _rules_summary(result):
    best = result['outcome_rules'].iloc[0]
    return {'outcome_rules': len(result['outcome_rules']), 'best_rule': rules.describe_rule(best),
            'best_lift': _r(best['lift']), 'best_confidence': _r(best['confidence'])}


def run_pipeline(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, figures_dir=FIGURES_DIR,
                 make_figures=True, verbose=True):
    """Run every step of the analysis and return the key metrics as a dict."""
    log = print if verbose else (lambda *args, **kwargs: None)

    def figure(plot_fn, *args):
        if make_figures:
            plot_fn(*args, out_dir=figures_dir)
            plt.close('all')

    if make_figures:
        plots.set_style()

    log('[1/7] Cleaning the data')
    cleaned = data.clean_all(*data.load_raw(raw_dir), verbose=False)
    data.save_processed(*cleaned, processed_dir=processed_dir)
    df_mat, df_por, df_xapi, df_both = data.load_processed(processed_dir)

    log('[2/7] Engineering features')
    df_mat_p = features.preprocess_uci(df_mat)
    df_por_p = features.preprocess_uci(df_por)
    df_both_p = features.add_dual_subject_features(df_both)
    df_x = features.preprocess_xapi(df_xapi)
    figure(plots.fig1_grade_distributions, df_mat_p, df_por_p)
    figure(plots.fig2_cross_subject, df_both_p)
    figure(plots.fig3_feature_distributions, df_mat_p, df_por_p)
    figure(plots.fig4_correlation_heatmaps, df_mat_p, df_por_p)
    figure(plots.fig5_xapi_eda, df_x)

    log('[3/7] Training Random Forest and Gradient Boosting')
    X_mat, y_mat, cols_mat = features.build_features(df_mat_p)
    X_por, y_por, cols_por = features.build_features(df_por_p)
    X_xapi, y_xapi, cols_xapi = features.xapi_features(df_x)
    res_math = models.train_random_forest(X_mat, y_mat)
    res_por = models.train_random_forest(X_por, y_por)
    res_gb = models.train_xapi_gb(X_xapi, y_xapi)
    figure(plots.fig6_rf_comparison, res_math, res_por)
    figure(plots.fig7_feature_importance,
           models.feature_importances(res_math['model'], cols_mat),
           models.feature_importances(res_por['model'], cols_por))
    figure(plots.fig8_gb_xapi, res_gb, cols_xapi)
    figure(plots.fig9_all_models_summary, res_math, res_por, res_gb)

    log('[4/7] Baselines and early-warning models')
    baselines, early, early_fitted = {}, {}, {}
    for subj, X, y in [('Math', X_mat, y_mat), ('Portuguese', X_por, y_por)]:
        baselines[subj] = models.compare_baselines(X, y)
        early[subj], early_fitted[subj] = models.early_warning_models(X, y)
    figure(plots.fig14_baselines_early_warning, baselines, early)

    log('[5/7] SHAP explanations')
    rf_early, X_te, y_te = early_fitted['Portuguese']
    shap_fail = explain.shap_for_fail(rf_early, X_te)
    drivers = explain.top_drivers(shap_fail, X_te)
    flagged = explain.flagged_students(rf_early, X_te, y_te)
    figure(plots.fig15_shap_summary, shap_fail)
    figure(plots.fig16_shap_single_student, shap_fail, flagged['index'],
           flagged['fail_prob'][flagged['index']])

    log('[6/7] Clustering and association rules')
    uci_cl = clustering.cluster_uci(df_mat_p, df_por_p)
    xapi_cl = clustering.cluster_xapi(df_x)
    xapi_arm = rules.xapi_rules(df_x)
    uci_arm = rules.uci_rules(df_mat_p)
    figure(plots.fig10_uci_clustering, uci_cl)
    figure(plots.fig11_xapi_clustering, xapi_cl)
    figure(plots.fig12_association_rules, xapi_arm['outcome_rules'], uci_arm['outcome_rules'])

    log('[7/7] Statistical tests')
    paired = stats_tests.paired_subject_ttest(df_both_p)
    cross = stats_tests.cross_subject_correlation(df_both_p)
    engagement = stats_tests.engagement_ttest(df_x)
    kruskal = stats_tests.engagement_kruskal(df_x)
    figure(plots.fig13_studytime_vs_grade, df_mat_p, df_por_p)

    return {
        'environment': {'python': platform.python_version(),
                        'scikit-learn': sklearn.__version__,
                        'random_state': RANDOM_STATE},
        'data': {'math_students': len(df_mat), 'portuguese_students': len(df_por),
                 'dual_enrolled_students': len(df_both), 'xapi_students': len(df_xapi)},
        'models': {'math_random_forest': _model_summary(res_math),
                   'portuguese_random_forest': _model_summary(res_por),
                   'xapi_gradient_boosting': _model_summary(res_gb)},
        'baselines': {subj.lower(): _results_table(df) for subj, df in baselines.items()},
        'early_warning': {subj.lower(): _results_table(df) for subj, df in early.items()},
        'shap_top_drivers_portuguese': {row.feature: _r(row.mean_abs_shap) for row in drivers.itertuples()},
        'clustering': {name: {'silhouette': _r(cl['silhouette']), 'davies_bouldin': _r(cl['davies_bouldin'])}
                       for name, cl in [('uci', uci_cl), ('xapi', xapi_cl)]},
        'association_rules': {'xapi': _rules_summary(xapi_arm), 'uci': _rules_summary(uci_arm)},
        'statistics': {
            'paired_ttest_math_vs_portuguese': {'t': _r(paired['t']), 'p': _sig(paired['p']),
                                                'mean_math': _r(paired['mean_math']),
                                                'mean_portuguese': _r(paired['mean_port'])},
            'correlation_math_vs_portuguese': {'r': _r(cross['r']), 'p': _sig(cross['p'])},
            'ttest_engagement_high_vs_low': {'t': _r(engagement['t']), 'p': _sig(engagement['p'])},
            'kruskal_engagement_by_class': {'H': _r(kruskal['H']), 'p': _sig(kruskal['p'])},
        },
    }


def print_summary(metrics):
    """Print the headline numbers."""
    print('\nKey results')
    names = {'math_random_forest': 'Math RF', 'portuguese_random_forest': 'Portuguese RF',
             'xapi_gradient_boosting': 'xAPI GB (3-class)'}
    for key, label in names.items():
        m = metrics['models'][key]
        print(f"  {label:<18} accuracy {m['accuracy']*100:.2f}%  AUC {m['auc']:.4f}  "
              f"5-fold CV {m['cv_mean']*100:.2f}% +/- {m['cv_std']*100:.2f}%")
    for subj in ['math', 'portuguese']:
        rf = metrics['early_warning'][subj]
        print(f"  Early warning ({subj}): AUC no grades {rf['Random Forest']['auc']:.3f}, "
              f"G1 only {rf['Random Forest (G1 only)']['auc']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Run the student performance analysis end to end.')
    parser.add_argument('--figures-dir', type=Path, default=FIGURES_DIR, help='where to save the figures')
    parser.add_argument('--metrics', type=Path, default=REPORTS_DIR / 'metrics.json',
                        help='where to save the metrics (JSON)')
    parser.add_argument('--no-figures', action='store_true', help='skip the figures (faster)')
    args = parser.parse_args()

    metrics = run_pipeline(figures_dir=args.figures_dir, make_figures=not args.no_figures)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')

    print_summary(metrics)
    if not args.no_figures:
        print(f'\nFigures saved to {args.figures_dir}')
    print(f'Metrics saved to {args.metrics}')


if __name__ == '__main__':
    main()
