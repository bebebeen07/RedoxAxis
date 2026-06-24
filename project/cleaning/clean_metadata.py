import logging
from pathlib import Path
from typing import List

import pandas as pd


def setup_logger() -> logging.Logger:
    """Create a logger for cleaning steps."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names by stripping whitespace and lowercasing.

    Parameters:
        df: Input DataFrame with metadata columns.

    Returns:
        DataFrame with cleaned column names.
    """
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    logger.info(f"Cleaned column names: {df.columns.tolist()}")
    return df


def deduplicate_by_entry(df: pd.DataFrame, entry_col: str = "Entry") -> pd.DataFrame:
    """Deduplicate rows by primary key Entry, preserving the first occurrence.

    Parameters:
        df: Input metadata DataFrame.
        entry_col: Column name for the UniProt primary key.

    Returns:
        Deduplicated DataFrame.
    """
    before = len(df)
    df = df.copy()
    df[entry_col] = df[entry_col].astype(str).str.strip()
    df = df.drop_duplicates(subset=[entry_col], keep="first")
    after = len(df)
    logger.info(f"Deduplicated by {entry_col}: {before} -> {after}")
    return df


def normalize_gene_symbol(gene_name: str) -> str:
    """Preserve the primary gene symbol from a possibly composite gene name field."""
    if pd.isna(gene_name):
        return gene_name
    gene_name = str(gene_name).strip()
    primary = gene_name.split(";")[0].split("/")[0].split(",")[0].strip()
    return primary


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Apply baseline cleaning to a metadata DataFrame.

    Parameters:
        df: Raw metadata DataFrame.

    Returns:
        Cleaned DataFrame with standardized column names and gene/protein fields.
    """
    df = clean_column_names(df)

    if "Entry" not in df.columns:
        raise ValueError("Metadata must contain an 'Entry' column as primary key.")

    if "Gene name" in df.columns:
        df["Gene name"] = df["Gene name"].apply(normalize_gene_symbol)
        logger.info("Normalized Gene name field")

    if "Protein name" in df.columns:
        df["Protein name"] = (
            df["Protein name"].astype(str)
            .str.replace(r"\s*\([^)]*\)", "", regex=True)
            .str.strip()
        )
        logger.info("Cleaned Protein name field")

    df = deduplicate_by_entry(df, entry_col="Entry")
    return df


def load_and_clean_metadata(file_path: str) -> pd.DataFrame:
    """Load a metadata file and run baseline cleaning with robust error handling.

    Parameters:
        file_path: Path to metadata CSV or Excel.

    Returns:
        Cleaned metadata DataFrame, or empty DataFrame if loading fails.
    """
    from utils.safe_io import safe_read_metadata
    path = Path(file_path)
    df, status = safe_read_metadata(path)

    if df is None:
        logger.error(f"Failed to load metadata from {path.name}: {status}")
        return pd.DataFrame()

    logger.info(f"Loaded metadata from {path}")
    return clean_metadata(df)


def clean_and_merge_metadata_files(file_paths: list) -> pd.DataFrame:
    """Load, clean, and merge multiple metadata files safely.

    Parameters:
        file_paths: List of paths to metadata files.

    Returns:
        Merged and cleaned DataFrame, or empty if no files successfully loaded.
    """
    cleaned_dfs = []
    for file_path in file_paths:
        df = load_and_clean_metadata(file_path)
        if not df.empty:
            cleaned_dfs.append(df)
        else:
            logger.warning(f"Skipped metadata file due to load/clean failure: {file_path}")

    if not cleaned_dfs:
        logger.warning("No metadata files were successfully loaded")
        return pd.DataFrame()

    merged = cleaned_dfs[0]
    for df in cleaned_dfs[1:]:
        merged = merged.merge(df, on="Entry", how="outer", suffixes=("", "_other"))

    logger.info(f"Merged {len(cleaned_dfs)} metadata files into shape {merged.shape}")

    # Coalesce duplicate "_other" columns back into the primary column so proteins
    # appearing only in files 2+ retain their values (e.g. EC number, Cofactor, etc.)
    # Must use positional indexing because duplicate column names cause df[name] → DataFrame.
    primary_cols = [c for c in cleaned_dfs[0].columns if c != "Entry"]
    col_list = merged.columns.tolist()
    for col in primary_cols:
        other_name = f"{col}_other"
        other_positions = [i for i, c in enumerate(col_list) if c == other_name]
        for pos in other_positions:
            merged[col] = merged[col].combine_first(merged.iloc[:, pos])
    # Drop all _other columns in one pass
    merged = merged.loc[:, [c for c in merged.columns if not str(c).endswith("_other")]]
    logger.info(f"After coalescing _other columns: {merged.shape}")

    # DEBUG: EC column audit after coalesce
    ec_related = [c for c in merged.columns if "ec" in str(c).lower() and "number" in str(c).lower()]
    print(f"[DEBUG clean_and_merge] EC-number columns after coalesce: {ec_related}")
    if "EC number" in merged.columns:
        missing_ec = merged["EC number"].isna().sum()
        print(f"[DEBUG clean_and_merge] Rows with NaN EC number: {missing_ec} / {len(merged)} ({missing_ec/len(merged)*100:.1f}%)")

    return merged
