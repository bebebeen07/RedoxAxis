import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def setup_logger() -> logging.Logger:
    """Create a logger for feature QC."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


def calculate_feature_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate sparsity and variance for each numeric feature.

    Parameters:
        df: Feature matrix DataFrame.

    Returns:
        Dictionary mapping feature names to their statistics.
    """
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        missing_ratio = df[col].isna().sum() / len(df)
        variance = df[col].var()
        stats[col] = {"sparsity": missing_ratio, "variance": variance}
    return stats


def detect_constant_features(df: pd.DataFrame, variance_threshold: float = 1e-6) -> List[str]:
    """Identify features with near-zero variance.

    Parameters:
        df: Feature matrix DataFrame.
        variance_threshold: Variance threshold below which features are considered constant.

    Returns:
        List of constant feature names.
    """
    constant_features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        variance = df[col].var()
        if pd.isna(variance) or variance < variance_threshold:
            constant_features.append(col)
    return constant_features


def detect_high_correlation_features(
    df: pd.DataFrame, correlation_threshold: float = 0.95
) -> List[Tuple[str, str, float]]:
    """Identify pairs of highly correlated features.

    Parameters:
        df: Feature matrix DataFrame.
        correlation_threshold: Correlation threshold.

    Returns:
        List of tuples (feature1, feature2, correlation).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= correlation_threshold:
                high_corr_pairs.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                )

    return high_corr_pairs


def perform_feature_qc(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Perform comprehensive feature quality control.

    Parameters:
        df: Feature matrix DataFrame.
        verbose: Whether to log QC results.

    Returns:
        Tuple of (cleaned DataFrame, QC report dictionary).
    """
    logger.info("Starting feature quality control")

    qc_report = {
        "initial_shape": df.shape,
        "dropped_features": [],
        "constant_features": [],
        "high_corr_pairs": [],
    }

    stats = calculate_feature_statistics(df)
    constant_features = detect_constant_features(df)
    high_corr_pairs = detect_high_correlation_features(df)

    qc_report["constant_features"] = constant_features
    qc_report["high_corr_pairs"] = high_corr_pairs

    df_clean = df.copy()

    if constant_features:
        df_clean = df_clean.drop(columns=constant_features)
        qc_report["dropped_features"].extend(constant_features)
        logger.warning(f"Dropped {len(constant_features)} constant features: {constant_features}")

    qc_report["final_shape"] = df_clean.shape
    high_sparsity_features = [col for col, stat in stats.items() if stat["sparsity"] > 0.9]
    qc_report["high_sparsity_features"] = high_sparsity_features

    if verbose:
        logger.info(f"Feature QC Report:")
        logger.info(f"  Initial shape: {qc_report['initial_shape']}")
        logger.info(f"  Final shape: {qc_report['final_shape']}")
        logger.info(f"  Dropped constant features: {len(constant_features)}")
        logger.info(f"  High sparsity features (>90%): {len(high_sparsity_features)}")
        logger.info(f"  Highly correlated pairs (>0.95): {len(high_corr_pairs)}")

    return df_clean, qc_report


# ─────────────────────────────────────────────────────────────────────────────
# Final Feature Matrix QC / Filtering
# ─────────────────────────────────────────────────────────────────────────────

# Prefixes to delete (unless the column also contains "_score")
_DROP_PREFIXES = ("catalytic_", "pathway_", "tissue_", "function_")

# Substrings that flag a column name as dirty
_DIRTY_KEYWORDS = (
    "evidence", "xref", "rhea", "pubmed", "chebi",
    "physiologicaldirection", "unknown_rec", "rulebase", "eco",
)

# Patterns that unconditionally protect a column from every deletion rule
_KEEP_PATTERNS = [
    re.compile(r"^ProteinEntry$"),
    re.compile(r"^(Entry|Protein_ID)$"),
    re.compile(r"^(Gene\s*Names?|Protein\s*names?|Length)$", re.IGNORECASE),
    re.compile(r"^aa_"),
    re.compile(
        r"^(sequence_length|molecular_weight|aromaticity|hydrophobicity"
        r"|instability_index|isoelectric_point|gravy)$"
    ),
    re.compile(r"^charge_"),
    re.compile(r"^ec_EC:"),
    re.compile(r"_score$"),
    re.compile(r"^semantic_emb_\d+$"),
]


def _is_protected(col: str) -> bool:
    return any(p.search(col) for p in _KEEP_PATTERNS)


def _drop_by_prefix(col: str) -> bool:
    """True if col should be deleted by the prefix rule."""
    col_lower = col.lower()
    if "_score" in col_lower:
        return False
    return any(col_lower.startswith(pfx) for pfx in _DROP_PREFIXES)


def _drop_by_keyword(col: str) -> bool:
    """True if col contains a dirty keyword."""
    col_lower = col.lower()
    return any(kw in col_lower for kw in _DIRTY_KEYWORDS)


def clean_final_feature_matrix(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
) -> pd.DataFrame:
    """Remove dirty, redundant, and zero-variance columns from the final feature matrix.

    Rules (in order):
      1. Duplicate column names → keep first occurrence
      2. Prefix-based deletion: catalytic_/ pathway_/ tissue_/ function_
         (columns that also contain '_score' are exempted)
      3. Dirty-keyword deletion (Evidence, xref, Rhea, PubMed, ChEBI, ECO, …)
      4. All-zero numeric columns
      5. Columns with >99 % missing values
    Protected columns are never deleted regardless of any rule.
    Writes a report to stdout and saves dropped column names to
    <output_dir>/dropped_features.txt.
    """
    output_dir = Path(output_dir)
    original_shape = df.shape
    dropped: List[str] = []

    # ── 1. Deduplicate column names ──────────────────────────────────────────
    dup_mask = df.columns.duplicated(keep="first")
    dup_cols = df.columns[dup_mask].tolist()
    if dup_cols:
        df = df.loc[:, ~dup_mask]
        dropped.extend(dup_cols)
        logger.info(f"Dropped {len(dup_cols)} duplicate columns")

    # ── 2. Prefix-based deletion ─────────────────────────────────────────────
    prefix_drop = [
        c for c in df.columns if not _is_protected(c) and _drop_by_prefix(c)
    ]
    df = df.drop(columns=prefix_drop)
    dropped.extend(prefix_drop)
    logger.info(f"Dropped {len(prefix_drop)} columns by prefix rule")

    # ── 3. Dirty-keyword deletion ────────────────────────────────────────────
    keyword_drop = [
        c for c in df.columns if not _is_protected(c) and _drop_by_keyword(c)
    ]
    df = df.drop(columns=keyword_drop)
    dropped.extend(keyword_drop)
    logger.info(f"Dropped {len(keyword_drop)} columns by keyword rule")

    # ── 4. All-zero numeric columns ──────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_cols = [c for c in numeric_cols if not _is_protected(c) and df[c].sum() == 0]
    df = df.drop(columns=zero_cols)
    dropped.extend(zero_cols)
    logger.info(f"Dropped {len(zero_cols)} all-zero columns")

    # ── 5. >99 % missing ────────────────────────────────────────────────────
    missing_rate = df.isnull().mean()
    high_missing = [
        c for c in df.columns
        if not _is_protected(c) and missing_rate[c] > 0.99
    ]
    df = df.drop(columns=high_missing)
    dropped.extend(high_missing)
    logger.info(f"Dropped {len(high_missing)} columns with >99% missing values")

    # ── Report ───────────────────────────────────────────────────────────────
    report_lines = [
        "",
        "=" * 60,
        "Feature Matrix QC Report",
        "=" * 60,
        f"  Original shape:        {original_shape}",
        f"  After cleaning:        {df.shape}",
        f"  Dropped columns count: {len(dropped)}",
        f"  Remaining features:    {df.shape[1]}",
        "=" * 60,
    ]
    print("\n".join(report_lines))

    output_dir.mkdir(parents=True, exist_ok=True)
    dropped_path = output_dir / "dropped_features.txt"
    dropped_path.write_text("\n".join(dropped), encoding="utf-8")
    logger.info(f"Dropped feature list saved to {dropped_path}")

    logger.info(f"Feature matrix cleaning complete: {original_shape} → {df.shape}")
    return df


def check_protein_uniqueness(df: pd.DataFrame) -> Tuple[bool, int, int]:
    """Verify that each protein Entry appears only once.

    Parameters:
        df: DataFrame with Entry column.

    Returns:
        Tuple of (is_unique: bool, total_rows: int, duplicate_entries: int).
    """
    if "Entry" not in df.columns:
        logger.warning("No 'Entry' column found for uniqueness check")
        return False, len(df), 0

    total_rows = len(df)
    unique_entries = df["Entry"].nunique()
    duplicate_entries = total_rows - unique_entries

    if duplicate_entries == 0:
        logger.info(f"✓ Protein uniqueness check passed: {unique_entries} unique entries in {total_rows} rows")
        return True, total_rows, 0
    else:
        logger.warning(
            f"✗ Protein uniqueness check FAILED: {duplicate_entries} duplicate entries found"
        )
        return False, total_rows, duplicate_entries
