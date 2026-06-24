import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_text(value: str) -> str:
    if pd.isna(value):
        return ""
    return str(value).upper().replace("-", "").replace("(", "").replace(")", "")


def assign_cofactor_group_from_fields(
    protein_names: str,
    ec_number: str,
    function_text: str,
    pathway: str,
    cofactor: str,
) -> str:
    for source in [protein_names, ec_number + " " + function_text, pathway, cofactor]:
        normalized = normalize_text(source)
        if "NADPH" in normalized:
            return "NADPH"
        if "NADH" in normalized and "NADPH" not in normalized:
            return "NADH"
        if "NADP" in normalized and "NADPH" not in normalized:
            return "NADP"
        if "NAD" in normalized:
            return "NAD"
    return "UNKNOWN"


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Assigning cofactor groups using multi-source robust matching")
    df = df.copy()

    protein_names = df.get("Protein names", pd.Series("", index=df.index)).fillna("").astype(str)
    ec_number = df.get("EC number", pd.Series("", index=df.index)).fillna("").astype(str)
    function_text = df.get("Function", pd.Series("", index=df.index)).fillna("").astype(str)
    pathway = df.get("Pathway", pd.Series("", index=df.index)).fillna("").astype(str)
    cofactor = df.get("Cofactor", pd.Series("", index=df.index)).fillna("").astype(str)

    df["CofactorGroup"] = [
        assign_cofactor_group_from_fields(
            protein_names.iloc[i],
            ec_number.iloc[i],
            function_text.iloc[i],
            pathway.iloc[i],
            cofactor.iloc[i],
        )
        for i in range(len(df))
    ]

    print(df["CofactorGroup"].value_counts(dropna=False))
    unknown_ratio = len(df[df["CofactorGroup"] == "UNKNOWN"]) / max(len(df), 1)
    print("UNKNOWN ratio:", unknown_ratio)
    logger.info("Cofactor group assignment complete")
    return df


def warn_small_groups(df: pd.DataFrame, min_count: int = 50) -> pd.DataFrame:
    for group in ["NAD", "NADH", "NADP", "NADPH"]:
        count = len(df[df["CofactorGroup"] == group])
        if count < min_count:
            logger.warning(
                "Group %s has only %s proteins, keeping group but data may be small",
                group,
                count,
            )
    return df
