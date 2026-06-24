import logging
from pathlib import Path
from typing import Dict, List

from cleaning.clean_metadata import clean_and_merge_metadata_files
from cleaning.handle_missing_values import (
    fill_categorical,
    fill_multilabel,
    flag_missing_sequences,
    impute_numeric,
)
from cleaning.merge_datasets import merge_fasta_and_metadata
from cleaning.standardize_fields import standardize_fields
from comparative_analysis.group_assignment import assign_groups, warn_small_groups
from data_loading.load_data import load_fasta
from feature_engineering.feature_qc import (
    check_protein_uniqueness,
    clean_final_feature_matrix,
    perform_feature_qc,
)
from feature_engineering.semantic_features import build_semantic_features
from matrix.build_feature_matrix import build_feature_matrix, save_feature_matrix
from comparative_analysis.cofactor_groups import (
    save_comparative_dataset,
    save_group_feature_matrices,
    save_group_labels,
)
from utils.config import COFACTOR_GROUPS, EXCLUDED_FILES, EXCLUDE_KEYWORDS
from utils.logging_utils import setup_root_logger
from utils.safe_io import save_load_log

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"


def collect_data_files(data_dir: Path):
    fasta_exts = {".fasta", ".fa", ".faa"}
    metadata_exts = {".csv", ".xlsx"}
    extra_exts = {".txt", ".md", ".pdf", ".docx"}

    fasta_files: List[Path] = []
    metadata_files: List[Path] = []
    extra_documents: List[Path] = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for path in sorted(data_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        name = path.name
        lower_name = name.lower()
        if name in EXCLUDED_FILES or any(keyword in lower_name for keyword in EXCLUDE_KEYWORDS):
            continue
        suffix = path.suffix.lower()
        if suffix in fasta_exts:
            fasta_files.append(path)
        elif suffix in metadata_exts:
            metadata_files.append(path)
        elif suffix in extra_exts:
            extra_documents.append(path)

    return fasta_files, metadata_files, extra_documents


def print_data_summary(
    fasta_files: List[Path],
    metadata_files: List[Path],
    extra_documents: List[Path],
) -> None:
    print("==================================")
    print("Protein Analysis Pipeline Started")
    print("==================================")
    print()
    print(f"Found FASTA files: {len(fasta_files)}")
    print(f"Found Metadata files: {len(metadata_files)}")
    print(f"Found Extra documents: {len(extra_documents)}")
    print()

    if fasta_files:
        print("FASTA:")
        for path in fasta_files:
            print(f"- {path.name}")
        print()

    if metadata_files:
        print("Metadata:")
        for path in metadata_files:
            print(f"- {path.name}")
        print()

    print("Found extra documents:")
    if extra_documents:
        for path in extra_documents:
            print(f"- {path.name}")
    print()


def main() -> None:
    setup_root_logger()
    logger = logging.getLogger(__name__)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fasta_files, metadata_files, extra_documents = collect_data_files(DATA_DIR)
    print_data_summary(fasta_files, metadata_files, extra_documents)

    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in {DATA_DIR}")
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found in {DATA_DIR}")

    logger.info("Starting protein feature engineering pipeline")

    fasta_df = load_fasta(fasta_files)
    logger.info(f"FASTA records loaded: {len(fasta_df)}")
    # Build authoritative Entry→group map from FASTA source files before any merge
    fasta_group_map = fasta_df.set_index("Entry")["CofactorGroup"].to_dict()

    metadata_dfs = clean_and_merge_metadata_files(metadata_files)

    if metadata_dfs.empty:
        logger.error("No metadata files were successfully loaded. Pipeline cannot continue.")
        save_load_log(OUTPUT_DIR / "load_log.txt")
        raise ValueError("No valid metadata files available")

    merged_metadata = metadata_dfs
    merged_metadata = standardize_fields(merged_metadata)
    merged_metadata = assign_groups(merged_metadata)
    merged_metadata = warn_small_groups(merged_metadata, min_count=50)
    merged_metadata = fill_categorical(merged_metadata, ["Organism", "Pathway", "Tissue"])
    merged_metadata = fill_multilabel(merged_metadata, ["Pathway", "Function", "Tissue"])
    merged_metadata = impute_numeric(merged_metadata, ["Redox potential", "Length"])

    merged_df = merge_fasta_and_metadata(fasta_df, merged_metadata)

    # FASTA source file is authoritative for group assignment
    merged_df["CofactorGroup"] = merged_df["Entry"].astype(str).map(fasta_group_map).fillna("UNKNOWN")
    logger.info(f"After FASTA-metadata merge: {len(merged_df)} rows")

    print("\n=== CofactorGroup distribution (from FASTA source files) ===")
    print(merged_df["CofactorGroup"].value_counts(dropna=False))
    print()

    merged_df = flag_missing_sequences(merged_df)

    is_unique, total, duplicates = check_protein_uniqueness(merged_df)
    if not is_unique:
        logger.error(f"Found {duplicates} duplicate Entry values. Fixing...")
        merged_df = merged_df.drop_duplicates(subset=["Entry"], keep="first")
        logger.info(f"After deduplication: {len(merged_df)} unique proteins")

    output_path = OUTPUT_DIR / "feature_matrix.csv"
    feature_matrix = build_feature_matrix(merged_df)
    logger.info(f"Feature matrix shape before QC: {feature_matrix.shape}")

    feature_matrix_clean, qc_report = perform_feature_qc(feature_matrix, verbose=True)
    logger.info(f"Feature matrix shape after QC: {feature_matrix_clean.shape}")

    # Append semantic text features (axis scores + embeddings) without touching existing features
    semantic_df = build_semantic_features(merged_df)
    semantic_df = semantic_df.rename(columns={"Entry": "ProteinEntry"})
    feature_matrix_clean = feature_matrix_clean.merge(semantic_df, on="ProteinEntry", how="left")
    logger.info(f"Feature matrix shape after semantic features: {feature_matrix_clean.shape}")

    feature_matrix_clean = clean_final_feature_matrix(feature_matrix_clean, OUTPUT_DIR)

    save_feature_matrix(feature_matrix_clean, str(output_path))

    save_group_feature_matrices(
        feature_matrix_clean,
        merged_df[["Entry", "Cofactor", "CofactorGroup"]],
        OUTPUT_DIR,
        COFACTOR_GROUPS,
    )
    save_group_labels(merged_df[["Entry", "Cofactor", "CofactorGroup"]], OUTPUT_DIR)
    save_comparative_dataset(feature_matrix_clean, merged_df[["Entry", "Cofactor", "CofactorGroup"]], OUTPUT_DIR)

    qc_log_path = OUTPUT_DIR / "feature_qc_report.txt"
    with open(qc_log_path, "w", encoding="utf-8") as f:
        f.write("Feature Quality Control Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Initial matrix shape: {qc_report['initial_shape']}\n")
        f.write(f"Final matrix shape: {qc_report['final_shape']}\n")
        f.write(f"Features dropped (constant): {len(qc_report['dropped_features'])}\n")
        f.write(f"High sparsity features (>90%): {len(qc_report['high_sparsity_features'])}\n")
        f.write(f"Highly correlated feature pairs (>0.95): {len(qc_report['high_corr_pairs'])}\n")

    save_load_log(OUTPUT_DIR / "load_log.txt")
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
