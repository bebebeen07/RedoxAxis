import logging
from typing import List

import pandas as pd


def setup_logger() -> logging.Logger:
    """Create a logger for dataset merging steps."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


def merge_on_entry(dataframes: list[pd.DataFrame], how: str = "outer") -> pd.DataFrame:
    """Merge multiple metadata DataFrames on the primary key Entry with deduplication.

    Parameters:
        dataframes: List of DataFrames containing an Entry column.
        how: Type of merge. Default 'outer' to preserve all proteins.

    Returns:
        Merged DataFrame with union of all protein entries, deduplicated by Entry.
    """
    if not dataframes:
        raise ValueError("No dataframes provided for merging.")

    merged_df = dataframes[0].copy()
    if "Entry" not in merged_df.columns:
        raise ValueError("Each DataFrame must contain an 'Entry' column.")

    for df in dataframes[1:]:
        if "Entry" not in df.columns:
            raise ValueError("Each DataFrame must contain an 'Entry' column.")
        merged_df = merged_df.merge(df, on="Entry", how=how, suffixes=("", "_other"))
        logger.info(f"Merged next dataset, resulting shape {merged_df.shape}")

    merged_df = merged_df.groupby("Entry", as_index=False).agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    )
    logger.info(f"Deduplicated by Entry, final shape {merged_df.shape}")
    return merged_df


def merge_fasta_and_metadata(fasta_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Join FASTA sequence data with metadata, keeping all proteins.

    Parameters:
        fasta_df: DataFrame produced from FASTA parsing.
        metadata_df: Combined metadata DataFrame.

    Returns:
        DataFrame containing sequence and metadata together, deduplicated by Entry.
    """
    merged_df = merge_on_entry([fasta_df, metadata_df], how="outer")
    logger.info(f"Merged FASTA and metadata into shape {merged_df.shape}")
    return merged_df
