from pathlib import Path


EXCLUDED_FILES = [
    "数据汇总.xlsx"
]

EXCLUDE_KEYWORDS = [
    "summary",
    "统计",
    "比例",
    "result",
    "analysis",
    "汇总"
]

COFACTOR_GROUPS = ["NAD", "NADH", "NADP", "NADPH"]


class PipelineConfig:
    """Configuration container for the protein feature engineering pipeline."""

    def __init__(self, fasta_path: str, metadata_paths: list[str], output_path: str):
        self.fasta_path = Path(fasta_path)
        self.metadata_paths = [Path(path) for path in metadata_paths]
        self.output_path = Path(output_path)

    def validate(self) -> None:
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_path}")
        for path in self.metadata_paths:
            if not path.exists():
                raise FileNotFoundError(f"Metadata file not found: {path}")
