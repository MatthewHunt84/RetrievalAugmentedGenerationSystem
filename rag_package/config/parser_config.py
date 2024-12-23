# config/parser_config.py
from dataclasses import dataclass, field
from typing import Literal, TypeAlias
from pathlib import Path
import logging

# Type definitions for parser configuration
ModelChoice: TypeAlias = Literal["sonnet_multimodal"]


@dataclass
class ParserConfig:
    """
    Configuration for the document parsing stage of the RAG pipeline.
    A list of all dependencies injected into the DocumentParser.

    Required Parameters:
        model: The model to use for parsing and understanding document structure
        use_cached_files: Whether to use cached parsing results if available

    Example:
        parsing_config = ParserConfig(
            model="sonnet_multimodal",
            use_cached_files=True
        )
    """

    # Required parameters
    model: ModelChoice
    use_cached_files: bool

    # Model configuration
    use_vendor_multimodal_model: bool = True
    vendor_multimodal_model_name: str = field(init=False)
    verbose: bool = True
    invalidate_cache: bool = True
    do_not_cache: bool = False

    # Parsing behavior
    result_type: str = "markdown"
    metadata_mode: str = "comprehensive"
    heading_detection: bool = True
    table_extraction: bool = True
    show_progress: bool = True
    num_workers: int = 8
    language: str = "en"

    # File paths
    image_dir: Path = Path("data_images")
    results_file: Path = Path("parsed_results.json")
    input_data_folder: Path = Path("raw_input_data")

    # Base instruction that applies to all models
    base_instruction: str = """
    You are processing technical product brochures for heavy machinery and equipment. 
    Please preserve:
    1. All technical specifications and measurements
    2. Model names and numbers with their release dates
    3. Hierarchical structure of the content (main sections, subsections)
    4. Detailed feature descriptions and capabilities
    5. Safety information and operational guidelines
    6. Any comparative information between models
    7. Manufacturing dates and historical information

    Format the content maintaining clear hierarchical structure using markdown:
    - Use # for main product categories
    - Use ## for individual models
    - Use ### for specification sections
    - Preserve all numerical values and units exactly as written
    - Maintain paragraph structure for detailed descriptions
    - Keep all technical terms in their original form
    """

    def __post_init__(self):
        """
        We can store models in the MODEL_VENDOR_MAPPING dictionary below
        to help make the main execution block a little more abstract and readable
        """
        self.vendor_multimodal_model_name = MODEL_VENDOR_MAPPING[self.model]

MODEL_VENDOR_MAPPING = {
    "sonnet_multimodal": "anthropic-sonnet-3.5"
}