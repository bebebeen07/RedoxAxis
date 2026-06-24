import logging
from typing import List

import pandas as pd


def setup_logger() -> logging.Logger:
    """Create a logger for tissue feature engineering."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


def parse_tissue_annotations(tissue_value: str) -> List[str]:
    """Parse tissue annotation strings into a normalized list of tissue labels."""
    if pd.isna(tissue_value) or not str(tissue_value).strip():
        return ["Unknown"]
    values = [part.strip().lower().replace(" ", "_") for part in str(tissue_value).split(";") if part.strip()]
    return sorted(values) if values else ["Unknown"]


def build_tissue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build tissue multi-hot encoded features from tissue annotation columns."""
    logger.info("Building tissue features")
    if "Tissue" not in df.columns:
        logger.warning("No Tissue column found for tissue feature extraction")
        return pd.DataFrame(index=df.index)

    rows = []
    for value in df["Tissue"].fillna("None"):
        labels = parse_tissue_annotations(value)
        row = {f"tissue_{label}": 1 for label in labels if label}
        rows.append(row)

    feature_df = pd.DataFrame(rows, index=df.index).fillna(0).astype(int)
    logger.info(f"Generated tissue feature matrix with shape {feature_df.shape}")
    return feature_df
