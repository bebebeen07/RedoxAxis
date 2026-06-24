import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Biological Axes – DO NOT modify axis names or keyword lists
# ─────────────────────────────────────────────────────────────────────────────
BIOLOGICAL_AXES = {
    "Metabolism_Bioenergetics": [
        "metabolism", "bioenergetics", "glycolysis", "tca cycle", "krebs cycle",
        "oxidative phosphorylation", "atp production", "carbohydrate degradation",
        "fatty acid beta-oxidation", "gluconeogenesis", "energy homeostasis",
        "pyruvate", "acetyl-coa", "metabolic pathway", "catabolism",
        "glucose homeostasis", "lipid metabolism", "amino acid metabolism",
        "dehydrogenase", "nutrient sensing",
    ],
    "Immune_Inflammation": [
        "immune", "inflammation", "inflammatory", "macrophage", "neutrophil",
        "t cell", "b cell", "leukocyte", "cytokine", "phagocytosis",
        "respiratory burst", "innate immunity", "adaptive immunity", "pathogen",
        "antimicrobial", "antibacterial", "antiviral", "nf-kappa-b",
        "toll-like receptor", "chemokine", "inflammasome", "lymphocyte activation",
    ],
    "Sleep_Circadian": [
        "circadian", "sleep", "rhythm", "clock", "diurnal", "bmal1", "per1",
        "per2", "clock gene", "melatonin", "suprachiasmatic nucleus",
        "chronobiology", "sleep-wake", "oscillation", "zeitgeber", "photoperiod",
        "sleep deprivation", "rhythmic expression", "biological clock",
        "light-dark cycle",
    ],
    "Redox_OxidativeStress": [
        "redox", "oxidative stress", "ros", "reactive oxygen species",
        "antioxidant", "glutathione", "thioredoxin", "hydrogen peroxide",
        "superoxide", "oxidative damage", "lipid peroxidation", "free radical",
        "nrf2", "oxidoreductase", "cellular stress", "hypoxia",
        "oxidative burst", "electron transfer",
    ],
    "Mitochondrial_Function": [
        "mitochondria", "mitochondrial", "complex i", "electron transport chain",
        "respiratory chain", "mitochondrial matrix", "inner mitochondrial membrane",
        "mitophagy", "mitochondrial dynamics", "mitochondrial fission",
        "mitochondrial fusion", "ubiquinone", "proton transport",
        "mitochondrial disease",
    ],
    "DNA_Repair_GenomicStability": [
        "dna repair", "genomic stability", "dna damage", "parp",
        "poly-adp-ribosylation", "base excision repair", "double-strand break",
        "homologous recombination", "non-homologous end joining",
        "chromatin remodeling", "dna replication", "mutagenesis", "uv damage",
        "genotoxic stress", "telomere maintenance",
    ],
    "Epigenetics_Transcription": [
        "epigenetic", "epigenetics", "transcription", "transcriptional regulation",
        "histone deacetylation", "sirtuin", "chromatin", "gene silencing",
        "promoter", "coactivator", "corepressor", "histone modification",
        "methylation", "acetylation", "dna methylation", "rna processing",
    ],
    "CellDeath_Aging": [
        "apoptosis", "cell death", "aging", "senescence", "longevity",
        "lifespan", "necrosis", "necroptosis", "ferroptosis", "caspase",
        "programmed cell death", "anti-aging", "cellular senescence",
        "autophagy", "survival pathway", "bcl-2", "p53", "cytochrome c release",
    ],
    "Calcium_Signaling": [
        "calcium signaling", "second messenger", "cadpr", "cyclic adp-ribose",
        "naadp", "cd38", "calcium release", "calcium homeostasis",
        "endoplasmic reticulum", "ryanodine receptor", "intracellular signaling",
        "signal transduction", "calmodulin", "ip3", "intracellular calcium",
    ],
    "Neurobiology": [
        "neurobiology", "neurogenesis", "neuron", "synaptic plasticity", "axon",
        "axonal degeneration", "neuroprotection", "neurodegenerative",
        "neurotransmitter", "dopamine", "serotonin", "brain", "hippocampus",
        "cortex", "myelination", "glial cell", "astrocyte", "microglia",
        "neurological", "cognitive",
    ],
    "Lipid_Steroid_Biosynthesis": [
        "lipid biosynthesis", "steroidogenesis", "steroid hormone",
        "cholesterol biosynthesis", "fatty acid synthesis", "lipogenesis",
        "adipogenesis", "estrogen", "androgen", "cortisol", "hormone metabolism",
        "triglyceride", "sphingolipid", "phospholipid", "mevalonate pathway",
        "sterol",
    ],
    "Xenobiotic_Metabolism": [
        "xenobiotic", "detoxification", "drug metabolism", "cytochrome p450",
        "cyp450", "toxin clearance", "liver metabolism", "hepatic",
        "phase i metabolism", "phase ii metabolism", "glucuronidation",
        "alcohol metabolism", "aldehyde dehydrogenase", "clearance",
        "biotransformation",
    ],
}

_MODEL_NAME = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _first_col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(_MODEL_NAME)
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for semantic features. "
            "Install with:  pip install sentence-transformers"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_semantic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 12 biological-axis scores + 384-dim embeddings for every protein.

    Parameters
    ----------
    df : merged_df with an 'Entry' column and the four text columns.

    Returns
    -------
    DataFrame with columns:
        Entry, <Axis>_score × 12, semantic_emb_1 … semantic_emb_384
    """
    from cleaning.text_normalizer import normalize_semantic_text_columns
    from sklearn.metrics.pairwise import cosine_similarity

    logger.info("Starting semantic feature engineering")

    # Normalize text columns (works on a copy, doesn't touch the caller's df)
    norm_df = normalize_semantic_text_columns(df)

    # Resolve actual column names (handles both original and standardized variants)
    pathway_col  = _first_col(norm_df, ["Pathway"])
    function_col = _first_col(norm_df, ["Function", "Function [CC]"])
    catalytic_col = _first_col(norm_df, ["Catalytic activity"])
    tissue_col   = _first_col(norm_df, ["Tissue", "Tissue specificity"])

    active_cols = [c for c in [pathway_col, function_col, catalytic_col, tissue_col] if c]
    logger.info(f"Semantic text columns used: {active_cols}")

    # Build one combined-text string per protein (vectorized)
    parts = [norm_df[c].fillna("").astype(str) for c in active_cols]
    combined: pd.Series = parts[0].copy()
    for p in parts[1:]:
        combined = combined + " " + p
    combined = combined.str.strip()
    # Fallback for fully-empty rows so the model always gets a non-empty string
    combined = combined.where(combined != "", "unknown protein")
    combined_texts = combined.tolist()

    # Load model and encode proteins
    logger.info(f"Loading SentenceTransformer model: {_MODEL_NAME}")
    model = _load_model()

    logger.info(f"Encoding {len(combined_texts)} protein texts …")
    protein_emb: np.ndarray = model.encode(
        combined_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    logger.info("Generated semantic embeddings")

    # Encode biological axes (join each keyword list into a single sentence)
    axis_names = list(BIOLOGICAL_AXES.keys())
    axis_texts  = [" ".join(kws) for kws in BIOLOGICAL_AXES.values()]
    axis_emb: np.ndarray = model.encode(
        axis_texts,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # Cosine similarity matrix: (n_proteins, n_axes), clipped to [0, 1]
    sims = cosine_similarity(protein_emb, axis_emb).astype(np.float32)
    sims = np.clip(sims, 0.0, 1.0)
    logger.info("Generated biological axis scores")

    # Assemble result DataFrame
    result = pd.DataFrame({"Entry": df["Entry"].astype(str).values})

    for i, name in enumerate(axis_names):
        result[f"{name}_score"] = sims[:, i]

    emb_dim = protein_emb.shape[1]  # 384 for all-MiniLM-L6-v2
    for dim in range(emb_dim):
        result[f"semantic_emb_{dim + 1}"] = protein_emb[:, dim].astype(np.float32)

    logger.info(
        f"Semantic feature matrix: {result.shape[0]} proteins × "
        f"{result.shape[1] - 1} features "
        f"({len(axis_names)} axis scores + {emb_dim} embedding dims)"
    )
    return result
