from pathlib import Path
from dataclasses import dataclass

@dataclass
class PathConfig:
    """Configuration for file paths used in node creation."""
    output_dir: Path = Path("node_outputs")
    analysis_dir: Path = Path("analysis")
    parsed_results_path: Path = Path("parsed_results.json")

    def __post_init__(self):
        """Ensure directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)