import logging
from typing import List

import pandas as pd


def setup_logger() -> logging.Logger:
    """Create a logger for pathway feature engineering."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()

PATHWAY_GROUPS = {
    "metabolism": ["metabolism", "glycolysis", "citric acid", "krebs", "oxidative phosphorylation"],
    "immune": ["immune", "inflammation", "cytokine", "antigen"],
    "signaling": ["signaling", "signal", "pathway", "receptor", "kinase"],
}


def normalize_pathway_labels(pathway_value: str) -> List[str]:
    """Split and normalize pathway annotations into canonical pathway groups."""
    if pd.isna(pathway_value) or not str(pathway_value).strip():
        return ["Unknown"]

    tokens = [part.strip().lower() for part in str(pathway_value).split(";") if part.strip()]
    groups = set()
    for token in tokens:
        for group, keywords in PATHWAY_GROUPS.items():
            if any(keyword in token for keyword in keywords):
                groups.add(group)
        if token not in groups:
            groups.add(token)
    return sorted(groups)


def build_pathway_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build pathway multi-hot encoded features from pathway annotations."""
    logger.info("Building pathway features")
    if "Pathway" not in df.columns:
        logger.warning("No Pathway column found for pathway feature extraction")
        return pd.DataFrame(index=df.index)

    rows = []
    for value in df["Pathway"].fillna("None"):
        labels = normalize_pathway_labels(value)
        row = {f"pathway_{label}": 1 for label in labels if label}
        rows.append(row)

    feature_df = pd.DataFrame(rows, index=df.index).fillna(0).astype(int)
    logger.info(f"Generated pathway feature matrix with shape {feature_df.shape}")
    return feature_df
