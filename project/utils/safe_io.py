import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def setup_safe_io_logger() -> logging.Logger:
    """Create a logger for safe I/O operations."""
    logger = logging.getLogger("safe_io")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_safe_io_logger()
LOAD_LOG_RECORDS = []


def add_load_log(status: str, filename: str, reason: str = "") -> None:
    """Record file loading status for export to log file."""
    record = f"[{status}] {filename}"
    if reason:
        record += f" - {reason}"
    LOAD_LOG_RECORDS.append(record)


def safe_read_excel(
    file_path: Path,
    nrows: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], str]:
    """Read an Excel file with fallback engine selection and error handling.

    Parameters:
        file_path: Path to Excel file.
        nrows: Maximum number of rows to read (protection against huge files).

    Returns:
        Tuple of (DataFrame or None, status message).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        msg = f"File not found: {file_path}"
        logger.error(msg)
        add_load_log("LOAD_FAILED", file_path.name, "File not found")
        return None, msg

    try:
        logger.info(f"Attempting to read Excel with calamine engine: {file_path.name}")
        df = pd.read_excel(file_path, engine="calamine", nrows=nrows)
        add_load_log("LOAD_SUCCESS", file_path.name, "calamine engine")
        logger.info(f"Successfully read {file_path.name} with calamine engine")
        return df, "Success (calamine)"
    except Exception as e:
        logger.warning(f"Calamine engine failed for {file_path.name}: {e}. Trying openpyxl...")

    try:
        logger.info(f"Attempting to read Excel with openpyxl engine: {file_path.name}")
        df = pd.read_excel(file_path, engine="openpyxl", nrows=nrows)
        add_load_log("LOAD_SUCCESS", file_path.name, "openpyxl engine (fallback)")
        logger.info(f"Successfully read {file_path.name} with openpyxl engine (fallback)")
        return df, "Success (openpyxl fallback)"
    except Exception as e:
        logger.warning(f"Openpyxl engine also failed for {file_path.name}: {e}. Trying xlrd...")

    try:
        logger.info(f"Attempting to read Excel with xlrd engine: {file_path.name}")
        df = pd.read_excel(file_path, engine="xlrd", nrows=nrows)
        add_load_log("LOAD_SUCCESS", file_path.name, "xlrd engine (fallback)")
        logger.info(f"Successfully read {file_path.name} with xlrd engine (final fallback)")
        return df, "Success (xlrd fallback)"
    except Exception as e:
        msg = f"All Excel engines failed for {file_path.name}: {e}"
        logger.error(msg)
        add_load_log("LOAD_FAILED", file_path.name, str(e)[:100])
        return None, msg


def safe_read_csv(
    file_path: Path,
    nrows: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], str]:
    """Read a CSV file with error handling.

    Parameters:
        file_path: Path to CSV file.
        nrows: Maximum number of rows to read.

    Returns:
        Tuple of (DataFrame or None, status message).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        msg = f"File not found: {file_path}"
        logger.error(msg)
        add_load_log("LOAD_FAILED", file_path.name, "File not found")
        return None, msg

    try:
        logger.info(f"Reading CSV file: {file_path.name}")
        df = pd.read_csv(file_path, nrows=nrows)
        add_load_log("LOAD_SUCCESS", file_path.name, "CSV")
        logger.info(f"Successfully read {file_path.name} with shape {df.shape}")
        return df, "Success"
    except Exception as e:
        msg = f"Failed to read CSV {file_path.name}: {e}"
        logger.error(msg)
        add_load_log("LOAD_FAILED", file_path.name, str(e)[:100])
        return None, msg


def safe_read_metadata(file_path: Path, nrows: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], str]:
    """Read a metadata file (CSV or Excel) with robust error handling.

    Parameters:
        file_path: Path to metadata file.
        nrows: Maximum number of rows to read.

    Returns:
        Tuple of (DataFrame or None, status message).
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return safe_read_csv(file_path, nrows=nrows)
    elif suffix in {".xls", ".xlsx"}:
        return safe_read_excel(file_path, nrows=nrows)
    else:
        msg = f"Unsupported metadata file extension: {suffix}"
        logger.error(msg)
        add_load_log("LOAD_SKIPPED", file_path.name, f"Unsupported extension: {suffix}")
        return None, msg


def get_load_log_records() -> list:
    """Return all load log records for export."""
    return LOAD_LOG_RECORDS


def save_load_log(output_path: Path) -> None:
    """Save load log records to a text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Data Loading Log\n")
        f.write("=" * 80 + "\n\n")
        for record in LOAD_LOG_RECORDS:
            f.write(record + "\n")
    logger.info(f"Saved load log to {output_path}")
