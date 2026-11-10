import numpy as np
import pandas as pd

from src import data


def test_cleaning_removes_duplicates():
    df = pd.DataFrame({'a': [1, 1, 2], 'b': ['x', 'x', 'y']})
    cleaned = data.clean_dataset(df, verbose=False)
    assert len(cleaned) == 2
    assert cleaned.duplicated().sum() == 0
    assert list(cleaned.index) == [0, 1]


def test_cleaning_fills_missing_values():
    df = pd.DataFrame({'num': [1.0, np.nan, 3.0, 5.0], 'text': ['a', 'b', None, 'a']})
    cleaned = data.clean_dataset(df, verbose=False)
    assert len(cleaned) == 4
    assert cleaned.isnull().sum().sum() == 0
    assert cleaned.loc[1, 'num'] == 3.0     # median of 1, 3, 5
    assert cleaned.loc[2, 'text'] == 'a'    # most common value


def test_cleaning_strips_spaces_and_keeps_input_unchanged():
    df = pd.DataFrame({'text': [' yes', 'no '], 'num': [1, 2]})
    cleaned = data.clean_dataset(df, verbose=False)
    assert cleaned['text'].tolist() == ['yes', 'no']
    assert df['text'].tolist() == [' yes', 'no ']


def test_raw_data_cleaning_matches_the_notebook():
    df_mat, df_por, df_xapi = data.load_raw()
    df_mat, df_por, df_xapi, df_both = data.clean_all(df_mat, df_por, df_xapi, verbose=False)
    assert df_mat.shape == (395, 33)
    assert df_por.shape == (649, 33)
    assert df_xapi.shape == (478, 17)   # 2 duplicate rows removed from 480
    assert len(df_both) == 382


def test_quality_audit_counts_problems():
    df = pd.DataFrame({'G3': [5, 25, 25], 'name': ['a', 'b', 'b']})
    audit = data.quality_audit(df, 'toy', target_col='G3', expected_range=(0, 20), verbose=False)
    assert audit['duplicates'] == 1
    assert audit['out_of_range'] == 2
    assert audit['text_cols'] == 1
