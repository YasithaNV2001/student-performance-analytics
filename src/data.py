"""Load, audit and clean the three datasets (notebook sections 2 to 2.6)."""
from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR

# Columns that identify the same student in the Math and Portuguese files
MERGE_COLS = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
              'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet']

PROCESSED_FILES = {
    'math': 'student-mat-cleaned.csv',
    'portuguese': 'student-por-cleaned.csv',
    'xapi': 'xAPI-cleaned.csv',
    'both': 'student-merged-cleaned.csv',
}


def load_raw(raw_dir=RAW_DIR):
    """Read the raw Math, Portuguese and xAPI files; returns (df_mat, df_por, df_xapi)."""
    raw_dir = Path(raw_dir)
    df_mat = pd.read_csv(raw_dir / 'student-mat.csv', sep=';')
    df_por = pd.read_csv(raw_dir / 'student-por.csv', sep=';')
    df_xapi = pd.read_csv(raw_dir / 'xAPI-Edu-Data.csv')
    return df_mat, df_por, df_xapi


def merge_subjects(df_mat, df_por):
    """Link students who appear in both the Math and the Portuguese file."""
    return pd.merge(df_mat, df_por, on=MERGE_COLS, suffixes=('_math', '_port'))


def text_columns(df):
    """Names of the text (categorical) columns."""
    return df.select_dtypes(include=['object', 'string']).columns.tolist()


def quality_audit(df, name, target_col=None, expected_range=None, verbose=True):
    """Count rows, missing values, duplicates and column types, and check the target's range."""
    audit = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing': int(df.isnull().sum().sum()),
        'missing_cols': df.columns[df.isnull().any()].tolist(),
        'duplicates': int(df.duplicated().sum()),
        'numeric_cols': len(df.select_dtypes(include='number').columns),
        'text_cols': len(text_columns(df)),
    }
    target_line = None
    if target_col and target_col in df.columns:
        vals = df[target_col]
        if expected_range:
            lo, hi = expected_range
            audit['out_of_range'] = int(((vals < lo) | (vals > hi)).sum())
            target_line = (f'  Target ({target_col})     : min={vals.min()}, max={vals.max()}, '
                           f'mean={vals.mean():.2f}, out-of-range={audit["out_of_range"]}')
        else:
            audit['classes'] = sorted(vals.unique().tolist())
            target_line = f'  Target ({target_col}) classes: {audit["classes"]}'

    if verbose:
        sep = '=' * 55
        print(sep)
        print('  ' + name)
        print(sep)
        print(f'  Shape            : {audit["rows"]:>5} rows x {audit["columns"]:>2} columns')
        print(f'  Missing values   : {audit["missing"]}')
        if audit['missing'] > 0:
            print(f'    Affected cols  : {audit["missing_cols"]}')
        print(f'  Duplicate rows   : {audit["duplicates"]}')
        print(f'  Numeric columns  : {audit["numeric_cols"]}')
        print(f'  Categorical cols : {audit["text_cols"]}')
        if target_line:
            print(target_line)
        print()
    return audit


def clean_dataset(df, name='dataset', verbose=True):
    """Fill missing values (median or most common value), drop duplicate rows, strip spaces from text."""
    log = print if verbose else (lambda *args, **kwargs: None)
    df = df.copy()
    log(f"--- Cleaning {name} Dataset ---")

    missing = df.isnull().sum().sum()
    if missing > 0:
        log(f"  [!] Found {missing} missing values. Imputing with column mode/median...")
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    else:
        log("  [✓] No missing values found.")

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        log(f"  [!] Found {duplicates} duplicate rows. Dropping duplicates...")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        log("  [✓] No duplicate rows found.")

    for col in text_columns(df):
        df[col] = df[col].str.strip()

    log(f"  [✓] Final cleaned shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


def clean_all(df_mat, df_por, df_xapi, verbose=True):
    """Clean the three datasets, then rebuild the dual-subject merge from the cleaned files."""
    df_mat = clean_dataset(df_mat, 'UCI Math', verbose)
    df_por = clean_dataset(df_por, 'UCI Portuguese', verbose)
    df_xapi = clean_dataset(df_xapi, 'xAPI Behavioral', verbose)
    df_both = merge_subjects(df_mat, df_por)
    return df_mat, df_por, df_xapi, df_both


def save_processed(df_mat, df_por, df_xapi, df_both, processed_dir=PROCESSED_DIR):
    """Write the four cleaned tables to data/processed/."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for key, df in zip(['math', 'portuguese', 'xapi', 'both'], [df_mat, df_por, df_xapi, df_both]):
        df.to_csv(processed_dir / PROCESSED_FILES[key], index=False)


def load_processed(processed_dir=PROCESSED_DIR):
    """Read the cleaned tables; returns (df_mat, df_por, df_xapi, df_both)."""
    processed_dir = Path(processed_dir)
    return tuple(pd.read_csv(processed_dir / PROCESSED_FILES[key])
                 for key in ['math', 'portuguese', 'xapi', 'both'])
