"""Statistical tests and supporting statistics (notebook section 9)."""
import pandas as pd
from scipy import stats

CORRELATION_FEATURES = ['studytime', 'failures', 'absences', 'avg_alcohol',
                        'parent_edu_avg', 'G1', 'G2']


def paired_subject_ttest(df_both_p):
    """Paired t-test: do the same students score differently in Math and Portuguese?"""
    t, p = stats.ttest_rel(df_both_p['G3_math'], df_both_p['G3_port'])
    return {'t': t, 'p': p, 'n': len(df_both_p),
            'mean_math': df_both_p['G3_math'].mean(), 'mean_port': df_both_p['G3_port'].mean()}


def engagement_ttest(df_x):
    """Independent t-test: engagement score of High-class versus Low-class xAPI students."""
    high = df_x[df_x['Class'] == 'H']['engagement_score']
    low = df_x[df_x['Class'] == 'L']['engagement_score']
    t, p = stats.ttest_ind(high, low)
    return {'t': t, 'p': p, 'mean_high': high.mean(), 'mean_low': low.mean()}


def grade_correlations(df, columns=CORRELATION_FEATURES, target='G3'):
    """Pearson r and p-value between each column and the final grade."""
    rows = []
    for col in columns:
        if col in df.columns:
            r, p = stats.pearsonr(df[col], df[target])
            rows.append({'feature': col, 'r': r, 'p': p})
    return pd.DataFrame(rows)


def cross_subject_correlation(df_both_p):
    """Pearson correlation between Math and Portuguese final grades."""
    r, p = stats.pearsonr(df_both_p['G3_math'], df_both_p['G3_port'])
    return {'r': r, 'p': p, 'n': len(df_both_p)}


def engagement_kruskal(df_x):
    """Kruskal-Wallis test: does engagement differ across the L, M and H classes?"""
    groups = [df_x[df_x['Class'] == c]['engagement_score'] for c in ['L', 'M', 'H']]
    h, p = stats.kruskal(*groups)
    return {'H': h, 'p': p}


def study_time_effect(df):
    """Mean final grade for students studying under 2 h/week versus 5 h or more."""
    low = df[df['studytime'] == 1]['G3'].mean()
    high = df[df['studytime'].isin([3, 4])]['G3'].mean()
    return {'under_2h': low, 'over_5h': high}


def high_low_ratio(df_x, column):
    """How many times larger the High-class mean of `column` is than the Low-class mean."""
    return (df_x[df_x['Class'] == 'H'][column].mean()
            / df_x[df_x['Class'] == 'L'][column].mean())
