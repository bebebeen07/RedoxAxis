import logging
from typing import Dict

import numpy as np
import pandas as pd
from Bio.SeqUtils import ProtParam


def setup_logger() -> logging.Logger:
    """Create a logger for sequence feature engineering."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
ALLOWED_AMINO_ACIDS = set(AMINO_ACIDS)


def clean_sequence(sequence: str) -> str:
    """Remove non-standard amino acids from a protein sequence."""
    if not isinstance(sequence, str):
        return ""
    cleaned = "".join([aa for aa in sequence.upper() if aa in ALLOWED_AMINO_ACIDS])
    return cleaned


def sequence_quality_check(sequence: str):
    """Evaluate sequence quality and return length, unknown ratio, and a quality label."""
    if not isinstance(sequence, str) or len(sequence) == 0:
        return 0, 0.0, "EMPTY"

    raw_sequence = sequence.upper()
    seq_length = len(raw_sequence)
    unknown_count = sum(1 for aa in raw_sequence if aa not in ALLOWED_AMINO_ACIDS)
    unknown_ratio = unknown_count / seq_length
    quality_label = "OK" if unknown_ratio < 0.1 and seq_length > 0 else "LOW_QUALITY"
    if seq_length == 0:
        quality_label = "EMPTY"
    return seq_length, unknown_ratio, quality_label


def calculate_sequence_features(sequence: str) -> Dict[str, float]:
    """Calculate amino acid composition and physicochemical sequence features."""
    if not sequence:
        return {f"aa_{aa}": 0.0 for aa in AMINO_ACIDS} | {
            "seq_length": 0,
            "cleaned_seq_length": 0,
            "molecular_weight": 0.0,
            "hydrophobicity": 0.0,
            "charge": 0.0,
            "aromaticity": 0.0,
            "complexity_score": 0.0,
            "sequence_invalid": 1,
            "sequence_quality": "EMPTY",
            "unknown_ratio": 0.0,
        }

    raw_length, unknown_ratio, quality_label = sequence_quality_check(sequence)
    cleaned_sequence = clean_sequence(sequence)
    cleaned_length = len(cleaned_sequence)

    if cleaned_length == 0:
        logger.warning(
            "Sequence cleaned to zero length; assigning fallback features and sequence_invalid flag"
        )
        return {f"aa_{aa}": 0.0 for aa in AMINO_ACIDS} | {
            "seq_length": raw_length,
            "cleaned_seq_length": 0,
            "molecular_weight": 0.0,
            "hydrophobicity": 0.0,
            "charge": 0.0,
            "aromaticity": 0.0,
            "complexity_score": 0.0,
            "sequence_invalid": 1,
            "sequence_quality": quality_label,
            "unknown_ratio": unknown_ratio,
        }

    try:
        protein = ProtParam.ProteinAnalysis(cleaned_sequence)
        aa_counts = protein.count_amino_acids()
        total = sum(aa_counts.get(aa, 0) for aa in AMINO_ACIDS) or 1
        composition = {f"aa_{aa}": aa_counts.get(aa, 0) / total for aa in AMINO_ACIDS}
        molecular_weight = protein.molecular_weight()
        hydrophobicity = protein.gravy()
        charge = protein.charge_at_pH(7.0)
        aromaticity = protein.aromaticity()
        complexity_score = np.log1p(cleaned_length) * (aromaticity + abs(hydrophobicity))
        sequence_invalid = 0
    except Exception as exc:
        logger.error(
            "Sequence feature calculation failed, returning fallback values: %s",
            exc,
        )
        composition = {f"aa_{aa}": 0.0 for aa in AMINO_ACIDS}
        molecular_weight = 0.0
        hydrophobicity = np.nan
        charge = np.nan
        aromaticity = np.nan
        complexity_score = np.nan
        sequence_invalid = 1

    logger.debug(
        "Sequence QC: raw_length=%d cleaned_length=%d unknown_ratio=%.3f quality=%s",
        raw_length,
        cleaned_length,
        unknown_ratio,
        quality_label,
    )

    return {
        **composition,
        "seq_length": raw_length,
        "cleaned_seq_length": cleaned_length,
        "molecular_weight": molecular_weight,
        "hydrophobicity": hydrophobicity,
        "charge": charge,
        "aromaticity": aromaticity,
        "complexity_score": complexity_score,
        "sequence_invalid": sequence_invalid,
        "sequence_quality": quality_label,
        "unknown_ratio": unknown_ratio,
    }


def build_sequence_feature_matrix(df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    """Generate per-protein sequence features from a DataFrame containing FASTA sequences."""
    logger.info("Building sequence feature matrix")
    features = df[sequence_col].fillna("").apply(calculate_sequence_features)
    feature_df = pd.DataFrame(features.tolist(), index=df.index)
    logger.info(f"Generated sequence features with shape {feature_df.shape}")
    return feature_df
