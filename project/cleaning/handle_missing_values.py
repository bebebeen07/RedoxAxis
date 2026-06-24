import logging
from typing import List

import pandas as pd


def setup_logger() -> logging.Logger:
    """Create a logger for missing value handling."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


def fill_categorical(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """Fill missing categorical columns with 'Unknown'."""
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            logger.info(f"Filled categorical missing values in column: {col}")
    return df


def fill_multilabel(df: pd.DataFrame, multilabel_cols: List[str]) -> pd.DataFrame:
    """Fill missing multi-label fields with 'None'."""
    df = df.copy()
    for col in multilabel_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")
            logger.info(f"Filled multi-label missing values in column: {col}")
    return df


def safe_numeric_impute(series: pd.Series, col_name: str):
    """Safely impute a numeric series and preserve a missing indicator.

    If the whole column is missing or non-numeric after coercion, skip median imputation.
    """
    numeric_series = pd.to_numeric(series, errors="coerce")
    indicator = numeric_series.isna().astype(int)

    if numeric_series.isna().all():
        logger.info(f"SKIP numeric column '{col_name}' because all values are missing or non-numeric")
        return numeric_series, indicator, False

    median_value = numeric_series.median(skipna=True)
    imputed_series = numeric_series.fillna(median_value)
    logger.info(
        f"IMPUTED numeric column '{col_name}' with median={median_value}, added indicator '{col_name}_is_missing'"
    )
    return imputed_series, indicator, True


def impute_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Impute numeric columns safely and add missing indicator columns."""
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            indicator_name = f"{col}_is_missing"
            imputed_series, indicator, imputed = safe_numeric_impute(df[col], col)
            df[indicator_name] = indicator
            df[col] = imputed_series
            if not imputed:
                logger.info(f"Kept numeric column '{col}' as-is with missing indicator '{indicator_name}'")
    return df


def flag_missing_sequences(df: pd.DataFrame, seq_col: str = "sequence") -> pd.DataFrame:
    """Flag missing FASTA sequence rows without dropping protein records."""
    df = df.copy()
    missing_flag = "missing_sequence"
    if seq_col in df.columns:
        df[missing_flag] = df[seq_col].isna().astype(int)
        df[seq_col] = df[seq_col].fillna("")
        logger.info(f"Flagged missing sequence in {missing_flag} column")
    return df
