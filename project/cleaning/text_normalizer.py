import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Candidate column names for each semantic slot (handles both original and standardized names)
_SLOT_CANDIDATES = {
    "pathway":   ["Pathway"],
    "function":  ["Function", "Function [CC]", "function [cc]"],
    "catalytic": ["Catalytic activity", "Catalytic"],
    "tissue":    ["Tissue", "Tissue specificity", "tissue specificity"],
}


def _normalize_single(text: str, is_catalytic: bool = False) -> str:
    """Apply all 9 normalization rules to one text value."""
    # Rule 9: NaN / empty / "None" → ""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    if not text or text.lower() in ("none", "nan", "unknown"):
        return ""

    # Rule 6: Remove evidence blocks  {ECO:0000256|...}
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"evidence\s*=\s*\{[^}]*\}", " ", text, flags=re.IGNORECASE)

    # Rule 7: Remove database cross-references
    text = re.sub(r"xref\s*=[^;]*",       " ", text, flags=re.IGNORECASE)
    text = re.sub(r"pubmed:\s*\d+",        " ", text, flags=re.IGNORECASE)
    text = re.sub(r"chebi:\s*(?:chebi:)?\s*\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"rhea-comp:\s*\S+",     " ", text, flags=re.IGNORECASE)
    text = re.sub(r"rhea:\s*(?:rhea:)?\s*\d+",   " ", text, flags=re.IGNORECASE)

    # Rule 8 (catalytic only): keep only "Reaction=..." content
    if is_catalytic:
        reactions = re.findall(r"reaction\s*=\s*([^;]+)", text, flags=re.IGNORECASE)
        if reactions:
            text = "; ".join(r.strip() for r in reactions)
        else:
            text = re.sub(r"physiologicaldirection\s*=[^;]*", " ", text, flags=re.IGNORECASE)

    # Rule 1: lowercase
    text = text.lower()

    # Rule 5: Remove special symbols – keep  - . +
    text = re.sub(r"[{}\[\]()<>\'\"=:]", " ", text)

    # Rule 4: Unify separators to "; "
    text = re.sub(r"[;|/,]+", "; ", text)

    # Rule 3: Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)

    # Rule 2: Strip
    return text.strip()


def normalize_semantic_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with Pathway/Function/Catalytic/Tissue normalized for embedding.

    Handles both original UniProt column names and the renamed forms produced by
    standardize_fields (e.g. 'Function [CC]' → 'Function', 'Tissue specificity' → 'Tissue').
    Other columns are passed through unchanged.
    """
    df = df.copy()
    processed = 0

    for slot, candidates in _SLOT_CANDIDATES.items():
        is_catalytic = (slot == "catalytic")
        for col in candidates:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x, c=is_catalytic: _normalize_single(x, is_catalytic=c)
                )
                processed += 1
                logger.debug(f"Normalized text column: '{col}'")
                break  # use first matching candidate only

    logger.info(f"Semantic text normalization completed ({processed} columns processed)")
    return df
