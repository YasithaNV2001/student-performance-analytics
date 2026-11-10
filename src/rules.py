"""Association rule mining with Apriori (notebook section 8)."""
from mlxtend.frequent_patterns import apriori, association_rules

from .features import BEHAV_COLS

XAPI_OUTCOMES = {'class_low', 'class_med', 'class_high'}
UCI_OUTCOMES = {'pass', 'fail'}


def mine_rules(transactions, min_support, min_confidence, outcomes):
    """Run Apriori, sort rules by lift, and keep the rules whose consequent is an outcome."""
    itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    rules = association_rules(itemsets, metric='confidence', min_threshold=min_confidence)
    rules = rules.sort_values('lift', ascending=False)
    outcome_rules = rules[rules['consequents'].apply(lambda x: bool(set(x) & outcomes))].copy()
    return {'itemsets': itemsets, 'rules': rules, 'outcome_rules': outcome_rules}


def xapi_transactions(df_x):
    """True/False items for xAPI: each behaviour at or above its median, absence, survey, class.

    Returns (transactions, median threshold per behaviour).
    """
    df = df_x.copy()
    thresholds = {}
    for col in BEHAV_COLS:
        thresholds[col] = df[col].median()
        df[f'{col}_high'] = (df[col] >= thresholds[col]).astype(bool)
    df['class_low'] = df['Class'] == 'L'
    df['class_med'] = df['Class'] == 'M'
    df['class_high'] = df['Class'] == 'H'
    df['absent_high'] = df['StudentAbsenceDays'] == 'Above-7'
    df['parent_yes'] = df['ParentAnsweringSurvey'] == 'Yes'
    items = ([f'{c}_high' for c in BEHAV_COLS]
             + ['absent_high', 'parent_yes', 'class_low', 'class_med', 'class_high'])
    return df[items].astype(bool), thresholds


def uci_transactions(df_p):
    """True/False items for a UCI dataset: study time, absences, failures, alcohol, plans, outcome."""
    df = df_p.copy()
    df['studytime_high'] = df['studytime'] >= df['studytime'].median()
    df['absences_high'] = df['absences'] >= df['absences'].median()
    df['failures_any'] = df['failures'] > 0
    df['alcohol_high'] = df['avg_alcohol'] >= df['avg_alcohol'].median()
    df['higher_edu'] = df['higher'] == 'yes'
    df['pass'] = df['pass_fail'] == 1
    df['fail'] = df['pass_fail'] == 0
    items = ['studytime_high', 'absences_high', 'failures_any', 'alcohol_high',
             'higher_edu', 'pass', 'fail']
    return df[items].astype(bool)


def xapi_rules(df_x, min_support=0.12, min_confidence=0.55):
    """Rules linking xAPI behaviour to the performance class."""
    transactions, thresholds = xapi_transactions(df_x)
    result = mine_rules(transactions, min_support, min_confidence, XAPI_OUTCOMES)
    result['thresholds'] = thresholds
    return result


def uci_rules(df_p, min_support=0.10, min_confidence=0.60):
    """Rules linking study habits and background to pass or fail."""
    return mine_rules(uci_transactions(df_p), min_support, min_confidence, UCI_OUTCOMES)


def describe_rule(rule):
    """Readable text for one rule, e.g. 'alcohol_high + failures_any -> fail'."""
    return f"{' + '.join(sorted(rule['antecedents']))} -> {' + '.join(sorted(rule['consequents']))}"
