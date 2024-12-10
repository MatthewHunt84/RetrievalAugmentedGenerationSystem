# rag_config.py
"""
RAG application settings module.

This module serves as a central configuration system for the RAG pipeline,
implementing a pseudo-singleton pattern for settings management. The configuration
is structured to handle the complex task of processing technical documentation
for heavy machinery and equipment.

Private variables are prefixed with an underscore: _private_variable
"""
from dataclasses import dataclass, field
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.core import Settings
import re

# Base directory for raw input data files
input_data_folder: str = "raw_input_data"

@dataclass
class ParserConfig:
    """
    Configuration for the initial parsing of technical documents. This class
    defines how raw documents should be processed and structured before entering
    the node creation phase.
    """
    result_type: str = "markdown"
    parsing_instruction: str = """
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
    use_vendor_multimodal_model: bool = True
    vendor_multimodal_model_name: str = "anthropic-sonnet-3.5"
    metadata_mode: str = "comprehensive"
    heading_detection: bool = True
    table_extraction_mode: str = "detailed"
    show_progress: bool = True
    verbose: bool = True
    invalidate_cache: bool = True
    do_not_cache: bool = False
    num_workers: int = 8
    language: str = "en"
    preserve_formatting: bool = True
    extract_tables: bool = True
    detect_lists: bool = True
    technical_terms_mode: str = "preserve"


# Global parsing configuration instance
use_cached_files: bool = True
parsing_config = ParserConfig()


@dataclass
class Measurement:
    """
    Represents a numerical measurement with its unit and context. Handles both
    single values and ranges while maintaining the original context.
    """
    value: float | int
    unit: str
    context: str | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        # Handle string inputs that might contain ranges
        if isinstance(self.value, str):
            if '-' in self.value:
                low, high = map(float, self.value.split('-'))
                self.value = (low + high) / 2  # Use average for now
                self.context = f"Range: {low}-{high} {self.unit}"
            else:
                self.value = float(self.value)


@dataclass
class EquipmentSpecifications:
    """
    Flexible specification structure for equipment metadata.
    """
    # Basic identification information
    basic_info: dict[str, str] = field(default_factory=dict)

    # Specifications with values and units
    specifications: dict[str, Measurement] = field(default_factory=dict)

    # Features and capabilities
    features: list[str] = field(default_factory=list)

    def add_specification(self, name: str, value: float | str, unit: str,
                          context: str | None = None) -> None:
        """Add a numerical specification with associated metadata."""
        spec_key = name.lower().replace(' ', '_')
        self.specifications[spec_key] = Measurement(value, unit, context)

    def add_feature(self, feature: str) -> None:
        """Add a descriptive feature."""
        normalized_feature = feature.lower().strip()
        if normalized_feature not in self.features:
            self.features.append(normalized_feature)

    def to_dict(self) -> dict:
        """Convert specifications to a serializable dictionary format."""
        return {
            "basic_info": self.basic_info,
            "specifications": {
                k: v.__dict__ for k, v in self.specifications.items()
            },
            "features": self.features
        }

@dataclass
class HierarchicalConfig:
    """
    Manages document hierarchy through header patterns and chunk sizing.
    Level 1 represents the highest/parent level (h1 headers),
    with deeper levels (2, 3) representing increasingly specific content.
    """
    # Map Markdown header levels to hierarchy levels
    header_hierarchy_map: dict[int, int] = field(default_factory=lambda: {
        1: 1,  # h1 headers (#) → top level 1 (parent)
        2: 2,  # h2 headers (##) → middle level 2
        3: 3,  # h3 headers (###) → lowest level 3
        4: 3,
        5: 3,
        6: 3
    })

    # Configure chunk sizes for different hierarchy levels
    chunk_sizes: dict[int, int] = field(default_factory=lambda: {
        1: 2048,  # Larger chunks for main sections (parents)
        2: 1024,  # Medium chunks for subsections
        3: 512   # Smaller chunks for detailed content
    })

    # Define overlap to maintain context between chunks
    chunk_overlaps: dict[int, int] = field(default_factory=lambda: {
        1: 256,  # Larger overlap for main sections
        2: 128,  # Medium overlap for subsections
        3: 64    # Smaller overlap for detailed content
    })

    def get_level_for_text(self, text: str) -> int:
        """
        Determine the hierarchy level of text based on its Markdown header depth.
        Returns 0 for non-header content.
        """
        first_line = text.strip().split('\n')[0]
        header_match = re.match(r'^(#+)\s', first_line)

        if not header_match:
            return 0

        header_level = len(header_match.group(1))
        return self.header_hierarchy_map.get(
            min(header_level, max(self.header_hierarchy_map.keys())),
            0
        )


@dataclass
class NodeCreationConfig:
    """Configuration for node creation."""
    pipeline_name: str = 'sixth_pipeline'
    parsed_results_path: str = 'parsed_results.json'
    output_dir: str = "node_outputs"

    # Keep the hierarchy config
    hierarchy_config: HierarchicalConfig = field(default_factory=HierarchicalConfig)

    @property
    def base_metadata(self) -> dict:
        """Provide base metadata structure for all nodes."""
        return {
            "pipeline_info": {
                "name": self.pipeline_name
            }
        }


# Global configuration instances
node_config = NodeCreationConfig()
embedding_model_name = "text-embedding-3-large"
multimodal_model = "claude-3-5-sonnet-latest"
multimodal_llm = AnthropicMultiModal(model=multimodal_model)

def get_embed_model() -> OpenAIEmbedding:
    """
    Singleton pattern for embedding model access.
    Ensures consistent embedding model usage across the application.
    """
    if not hasattr(get_embed_model, '_instance'):
        get_embed_model._instance = OpenAIEmbedding(model="text-embedding-3-large")
        Settings.embed_model = get_embed_model._instance
    return get_embed_model._instance