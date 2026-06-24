import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_cofactor(cofactor: str) -> str:
    if pd.isna(cofactor) or not str(cofactor).strip():
        return "Unknown"
    value = str(cofactor).strip().upper().replace("+", "").replace(" ", "")
    if value in {"NAD", "NADH", "NADP", "NADPH"}:
        return value
    if "NADP" in value:
        return "NADP"
    if "NADPH" in value:
        return "NADPH"
    if "NADH" in value:
        return "NADH"
    if "NAD" in value:
        return "NAD"
    return "Other"


def add_cofactor_group_labels(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Assigning CofactorGroup based on Cofactor values")
    df = df.copy()
    if "Cofactor" not in df.columns:
        df["Cofactor"] = "Unknown"
    df["CofactorGroup"] = df["Cofactor"].apply(normalize_cofactor)
    return df


def save_group_feature_matrices(
    feature_matrix: pd.DataFrame,
    metadata_labels: pd.DataFrame,
    output_dir: Path,
    groups: List[str],
) -> None:
    """Save separate feature matrices for each cofactor group."""
    output_dir = Path(output_dir)
    for group in groups:
        subset_ids = metadata_labels.loc[metadata_labels["CofactorGroup"] == group, "Entry"].astype(str)
        subset_matrix = feature_matrix[feature_matrix["ProteinEntry"].isin(subset_ids)]
        sink = output_dir / f"{group}_feature_matrix.csv"
        subset_matrix.to_csv(sink, index=False)
        logger.info(f"Saved feature matrix for group {group} to {sink}")


def save_group_labels(metadata_labels: pd.DataFrame, output_dir: Path) -> None:
    """Save group labels per protein entry."""
    output_dir = Path(output_dir)
    group_labels = metadata_labels.copy()
    group_labels["Entry"] = group_labels["Entry"].astype(str)
    sink = output_dir / "group_labels.csv"
    group_labels.to_csv(sink, index=False)
    logger.info(f"Saved protein group labels to {sink}")


def save_comparative_dataset(
    feature_matrix: pd.DataFrame,
    metadata_labels: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save a joined comparative dataset with group labels for ML training or analysis."""
    output_dir = Path(output_dir)
    joined = feature_matrix.merge(
        metadata_labels.rename(columns={"Entry": "ProteinEntry"}),
        on="ProteinEntry",
        how="left",
    )
    sink = output_dir / "comparative_dataset.csv"
    joined.to_csv(sink, index=False)
    logger.info(f"Saved comparative dataset to {sink}")
