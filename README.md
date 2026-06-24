# 🧬 RedoxAxis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" />
  <img src="https://img.shields.io/badge/Bioinformatics-Pipeline-green.svg" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Ready-orange.svg" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" />
</p>

<p align="center">
<b>An automated bioinformatics pipeline for large-scale redox enzyme analysis, semantic feature engineering, and cross-biological-axis modeling.</b>
</p>

---

## Overview

**RedoxAxis** is an end-to-end bioinformatics framework designed for the systematic analysis of redox-related proteins, particularly enzymes associated with **NAD, NADH, NADP, and NADPH**.

The pipeline automatically integrates heterogeneous biological datasets from UniProt, performs sequence and functional feature engineering, extracts semantic biological knowledge from textual annotations, and constructs machine-learning-ready feature matrices for downstream comparative analyses.

RedoxAxis aims to bridge traditional bioinformatics and modern AI by transforming free-text protein annotations into biologically meaningful quantitative representations.

---

## Core Features

### Automatic Data Discovery
- Automatically scans local directories for FASTA, Excel, CSV, TXT, and Markdown files.
- Supports batch processing of multiple datasets simultaneously.
- Automatically recognizes NAD/NADH/NADP/NADPH protein groups.

### Intelligent Data Cleaning
- Standardizes heterogeneous UniProt annotations.
- Removes duplicated protein entries.
- Handles missing values and inconsistent metadata formats.
- Robust preprocessing to avoid malformed feature columns.

### Protein Sequence Feature Engineering
- Sequence length statistics.
- Molecular weight calculation.
- Isoelectric point estimation.
- Amino acid composition profiling.
- Sequence complexity measurements.

### Functional Annotation Processing
- Multi-label EC number extraction.
- Hierarchical EC grouping.
- Catalytic activity parsing.
- Pathway annotation extraction.
- Tissue specificity analysis.

### Semantic Biological Feature Extraction
- Embedding-based representation learning for biological annotations.
- Converts free-text annotations into numerical vectors.
- Integrates semantic information from:
  - Function [CC]
  - Catalytic Activity
  - Pathway
  - Cofactor
  - Tissue Specificity

### Cross-Biological Axis Scoring
RedoxAxis projects protein functions onto multiple biological dimensions:

| Biological Axis | Description |
|-----------------|-------------|
| Metabolism & Bioenergetics | Energy metabolism and metabolic regulation |
| Immune & Inflammation | Immune responses and inflammatory processes |
| Sleep & Circadian | Circadian rhythm and sleep regulation |
| Redox & Oxidative Stress | Oxidative homeostasis and ROS biology |
| Mitochondrial Function | Mitochondrial activity and respiration |
| DNA Repair | Genome maintenance and DNA repair |
| Epigenetics | Chromatin and transcription regulation |
| Aging & Cell Death | Senescence and apoptosis |
| Neurobiology | Nervous system and cognitive function |
| Xenobiotic Metabolism | Drug and toxin metabolism |

---

# Recommended Project Structure

```text
ProjectRoot/
│
.
├── data/                          # User-created directory for raw inputs (see guide below)
├── project/                       # Core package directory
│   ├── cleaning/                  # Data preprocessing and standardization
│   │   ├── clean_metadata.py
│   │   ├── handle_missing_values.py
│   │   ├── merge_datasets.py
│   │   ├── standardize_fields.py
│   │   └── text_normalizer.py
│   ├── comparative_analysis/      # Group assignment and cross-axis profiling
│   │   ├── __init__.py
│   │   ├── cofactor_groups.py
│   │   └── group_assignment.py
│   ├── data_loading/              # Input file parsing and ingestion
│   │   └── load_data.py
│   ├── feature_engineering/       # Multi-modal feature extraction modules
│   │   ├── feature_qc.py
│   │   ├── functional_features.py
│   │   ├── pathway_features.py
│   │   ├── semantic_features.py
│   │   ├── sequence_features.py
│   │   └── tissue_features.py
│   ├── matrix/                    # Feature matrix compilation
│   │   └── build_feature_matrix.py
│   ├── utils/                     # Shared utility functions
│   └── main.py                    # Main pipeline execution entry point
└── README.md
```

## Data Preparation Guide

>  **IMPORTANT**
> This repository does NOT include any raw datasets.
> You must manually prepare the `data/` directory before running `project/main.py`.

---

### 1️. Create the data folder

In the **project root directory** (same level as `project/`), create a folder named:

```bash
data
```

### 2️. Add your datasets
Manually place or upload all biological source datasets into the `data/` folder you just created.

Your directory should look like:

```text
RedoxAxis/
├── project/
├── data/
│   ├── xxx.xlsx
│   ├── xxx.fasta
│   ├── xxx.csv
│   └── xxx.txt
```

### 3️. Supported file formats

The pipeline will automatically detect and parse the following file types:

---

#### Tabular data

```text
.xlsx   Excel files
.csv    Comma-separated tables
```

#### Sequence data

```text
.fasta  Protein / nucleotide sequences
```

#### Reference / annotation files

```text
.txt    Plain text files
.md     Markdown documents
```

### NOTE

Ensure your source files maintain uniform primary keys, such as UniProt Accession IDs, to guarantee accurate cross-dataset merging and cleaning.

