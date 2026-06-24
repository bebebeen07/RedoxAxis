import logging
from pathlib import Path
from typing import List

import pandas as pd

from feature_engineering.functional_features import build_functional_features
from feature_engineering.pathway_features import build_pathway_features
from feature_engineering.sequence_features import build_sequence_feature_matrix
from feature_engineering.tissue_features import build_tissue_features


def setup_logger() -> logging.Logger:
    """Create a logger for feature matrix construction."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the final ML-ready feature matrix from protein metadata and sequences."""
    logger.info("Building final feature matrix")

    sequence_features = build_sequence_feature_matrix(df)
    functional_features = build_functional_features(df)
    pathway_features = build_pathway_features(df)
    tissue_features = build_tissue_features(df)

    feature_frames = [sequence_features, functional_features, pathway_features, tissue_features]
    feature_matrix = pd.concat(feature_frames, axis=1).fillna(0)

    feature_matrix.index = df["Entry"].astype(str)
    feature_matrix = feature_matrix.reset_index(drop=False).rename(columns={"Entry": "ProteinEntry"})
    logger.info(f"Final feature matrix shape {feature_matrix.shape}")
    return feature_matrix


def save_feature_matrix(df: pd.DataFrame, output_path: str) -> None:
    """Save the feature matrix to CSV for downstream ML or analysis."""
    output_file = Path(output_path)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved feature matrix to {output_file}")
