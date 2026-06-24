import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from Bio import SeqIO

from utils.safe_io import safe_read_metadata


def setup_logger() -> logging.Logger:
    """Create a simple logger for the data loading module."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


def _group_from_filename(filename: str) -> str:
    name = filename.upper()
    if "NADPH" in name:
        return "NADPH"
    elif "NADH" in name:
        return "NADH"
    elif "NADP" in name:
        return "NADP"
    elif "NAD" in name:
        return "NAD"
    return "UNKNOWN"


def parse_fasta(fasta_path: Union[str, Path], source_group: str = "UNKNOWN") -> Optional[pd.DataFrame]:
    fasta_file = Path(fasta_path)
    logger.info(f"Loading FASTA sequences from {fasta_file}")

    try:
        records = []
        for record in SeqIO.parse(str(fasta_file), "fasta"):
            # 【修复】标准 UniProt 格式：sp|P12345|AATM_HUMAN
            # 切分后应该取中间的 [1] 索引，即 P12345
            if "|" in record.id:
                parts = record.id.split("|")
                entry = parts[1] if len(parts) > 1 else record.id
            else:
                entry = record.id

            # 标准化：去除可能存在的空格并转大写
            entry = str(entry).strip()

            sequence = str(record.seq).upper()
            records.append(
                {
                    "Entry": entry,
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "missing_sequence": 0,
                    "CofactorGroup": source_group,
                }
            )

        if not records:
            logger.warning(f"No sequences found in FASTA file: {fasta_file}")

        fasta_df = pd.DataFrame(records)
        logger.info(f"Parsed {len(fasta_df)} FASTA records from {fasta_file.name}")
        return fasta_df
    except Exception as e:
        logger.error(f"Failed to parse FASTA file {fasta_file.name}: {e}")
        return None


def load_fasta(fasta_paths: List[Path]) -> pd.DataFrame:
    print("\n=== FASTA file → group mapping ===")
    data_frames = []
    for path in fasta_paths:
        group = _group_from_filename(path.name)
        print(f"  {path.name} → {group}")
        df = parse_fasta(path, source_group=group)
        if df is not None:
            data_frames.append(df)
    print()

    if data_frames:
        # 【修复】合并后，必须根据唯一主键 'Entry' 进行去重，保留第一次出现的序列
        combined = pd.concat(data_frames, ignore_index=True)
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=["Entry"], keep="first")
        logger.info(f"FASTA concatenation deduplication: {before_dedup} -> {len(combined)}")
    else:
        combined = pd.DataFrame(columns=["Entry", "sequence", "sequence_length", "missing_sequence", "CofactorGroup"])

    print("=== FASTA group distribution ===")
    print(combined["CofactorGroup"].value_counts(dropna=False))
    print()
    logger.info(f"Successfully loaded and deduped FASTA files with shape {combined.shape}")
    return combined

def load_metadata_files(metadata_paths: List[Path]) -> List[pd.DataFrame]:
    """Load multiple metadata files with robust error handling.

    Parameters:
        metadata_paths: List of paths to metadata CSV or Excel files.

    Returns:
        List of successfully loaded DataFrames (may be shorter than input if some files fail).
    """
    loaded_dfs = []
    for path in metadata_paths:
        df, status = safe_read_metadata(path)
        if df is not None:
            loaded_dfs.append(df)
            logger.info(f"Loaded metadata from {path.name}: {status}")
        else:
            logger.warning(f"Skipped metadata file {path.name}: {status}")

    logger.info(f"Successfully loaded {len(loaded_dfs)} out of {len(metadata_paths)} metadata files")
    return loaded_dfs


def load_metadata(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load a single metadata table from CSV or Excel using safe I/O.

    Parameters:
        file_path: Path to metadata CSV or Excel file.

    Returns:
        DataFrame or None if loading fails.
    """
    file_path = Path(file_path)
    df, status = safe_read_metadata(file_path)
    if df is not None:
        logger.info(f"Loaded metadata '{file_path.name}' with shape {df.shape}")
    else:
        logger.error(f"Failed to load metadata '{file_path.name}': {status}")
    return df


def load_data(fasta_paths: List[Path], metadata_paths: List[Path]) -> Dict[str, Union[pd.DataFrame, list]]:
    """Load FASTA and metadata tables from lists of Paths with robust error handling.

    Parameters:
        fasta_paths: List of paths to FASTA files.
        metadata_paths: List of paths to metadata files.

    Returns:
        Dictionary with 'fasta' DataFrame and 'metadata' list of DataFrames.
    """
    fasta_df = load_fasta(fasta_paths)
    metadata_dfs = load_metadata_files(metadata_paths)
    return {"fasta": fasta_df, "metadata": metadata_dfs}
