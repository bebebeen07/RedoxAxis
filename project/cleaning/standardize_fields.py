import logging
from typing import Any, Dict

import pandas as pd


def setup_logger() -> logging.Logger:
    """Create a logger for standardization steps."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()

ORGANISM_MAP = {
    "9606": "Human",
    "homo sapiens": "Human",
    "human": "Human",
    "homo sapiens (human)": "Human",
}

PATHWAY_MAP = {
    "kegg": "KEGG",
    "reactome": "Reactome",
    "metabolism": "Metabolism",
    "immune": "Immune",
    "signaling": "Signaling",
}


def normalize_organism(value: Any) -> str:
    """Standardize organism labels to a consistent human-readable name."""
    if pd.isna(value):
        return "Unknown"
    value = str(value).strip().lower()
    return ORGANISM_MAP.get(value, value.title())


def normalize_ec_number(value: Any) -> str:
    """Standardize EC numbers to canonical EC:x.x.x.x format."""
    if pd.isna(value):
        return "Unknown"
    value = str(value).strip()
    if value.lower().startswith("ec:"):
        value = value[3:].strip()
    parts = [part.strip() for part in value.split(".") if part.strip()]
    if len(parts) == 4:
        return f"EC:{'.'.join(parts)}"
    return value


def normalize_pathway(value: Any) -> str:
    """Normalize pathway naming conventions to an agreed set of terms."""
    if pd.isna(value):
        return "Unknown"
    value = str(value).strip()
    lower = value.lower()
    for key, mapped in PATHWAY_MAP.items():
        if key in lower:
            return mapped
    return value


def normalize_tissue(value: Any) -> str:
    """Simple normalization of tissue labels; missing values become Unknown."""
    if pd.isna(value):
        return "Unknown"
    value = str(value).strip()
    return value.title()


def standardize_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Apply field normalization rules to organism, EC, pathway, and tissue columns."""
    df = df.copy()

    alias_map = {
        "Protein name": "Protein names",
        "Function [CC]": "Function",
        "function [cc]": "Function",
        "function[cc]": "Function",
        "Cofactors": "Cofactor",
        "Tissue specificity": "Tissue",
        "tissue specificity": "Tissue",
    }

    for original, target in alias_map.items():
        if original in df.columns and target not in df.columns:
            df = df.rename(columns={original: target})
            logger.info(f"Renamed '{original}' to '{target}'")

    if "Organism" in df.columns:
        df["Organism"] = df["Organism"].apply(normalize_organism)
        logger.info("Standardized Organism values")

    if "EC number" in df.columns:
        df["EC number"] = df["EC number"].apply(normalize_ec_number)
        logger.info("Standardized EC number values")

    if "Pathway" in df.columns:
        df["Pathway"] = df["Pathway"].apply(normalize_pathway)
        logger.info("Standardized Pathway values")

    if "Tissue" in df.columns:
        df["Tissue"] = df["Tissue"].apply(normalize_tissue)
        logger.info("Standardized Tissue values")

    return df

