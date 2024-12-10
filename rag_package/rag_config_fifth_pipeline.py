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
    Flexible specification structure for equipment metadata. Avoids rigid
    categorization while maintaining structured data where appropriate.
    """
    # Basic identification and classification information
    basic_info: dict[str, str] = field(default_factory=dict)

    # All numerical specifications in a flat structure for easy access
    numerical_specs: dict[str, Measurement] = field(default_factory=dict)

    # Extracted features and capabilities as normalized tags
    attribute_tags: list[str] = field(default_factory=list)

    # Original text chunks for context preservation
    text_chunks: list[str] = field(default_factory=list)

    # Track confidence scores for extracted information
    extraction_confidence: dict[str, float] = field(default_factory=dict)

    def add_specification(self, name: str, value: float | str, unit: str,
                          context: str | None = None, confidence: float = 1.0) -> None:
        """
        Add a numerical specification with associated metadata.
        Normalizes specification names for consistent access.
        """
        spec_key = name.lower().replace(' ', '_')
        self.numerical_specs[spec_key] = Measurement(value, unit, context, confidence)

    def add_tag(self, tag: str, confidence: float = 1.0) -> None:
        """
        Add a descriptive tag with confidence score.
        Normalizes tags and prevents duplicates.
        """
        normalized_tag = tag.lower().strip()
        if normalized_tag not in self.attribute_tags:
            self.attribute_tags.append(normalized_tag)
            self.extraction_confidence[normalized_tag] = confidence

    def to_dict(self) -> dict:
        """Convert specifications to a serializable dictionary format."""
        return {
            "basic_info": self.basic_info,
            "numerical_specs": {
                k: v.__dict__ for k, v in self.numerical_specs.items()
            },
            "attribute_tags": self.attribute_tags,
            "extraction_confidence": self.extraction_confidence
        }


@dataclass
class HierarchicalConfig:
    """
    Manages document hierarchy through header patterns and chunk sizing.
    Provides a consistent way to determine content hierarchy levels.
    """
    # Map Markdown header levels to hierarchy levels
    header_hierarchy_map: dict[int, int] = field(default_factory=lambda: {
        1: 3,  # h1 headers (single #) → highest level 3
        2: 2,  # h2 headers (##) → middle level 2
        3: 1,  # h3+ headers (### or more) → lowest level 1
        4: 1,
        5: 1,
        6: 1
    })

    # Configure chunk sizes for different hierarchy levels
    chunk_sizes: dict[int, int] = field(default_factory=lambda: {
        3: 1024,  # Larger chunks for main sections
        2: 512,  # Medium chunks for model details
        1: 256  # Smaller chunks for specifications
    })

    # Define overlap to maintain context between chunks
    chunk_overlaps: dict[int, int] = field(default_factory=lambda: {
        3: 128,  # ~12.5% overlap for continuity
        2: 64,
        1: 32
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
    """
    Configuration for node creation with flexible metadata extraction.
    Combines hierarchical processing with comprehensive metadata capture.

    This class maintains all necessary configuration parameters including:
    - Pipeline identification and paths
    - Hierarchical document processing settings
    - Patterns for extracting numerical values and features
    - Confidence thresholds for extraction quality
    """
    # Pipeline identification and basic paths
    pipeline_name: str = 'fifth_pipeline'
    parsed_results_path: str = 'parsed_results.json'  # Added this back
    output_dir: str = "node_outputs"

    # Core configuration components
    hierarchy_config: HierarchicalConfig = field(default_factory=HierarchicalConfig)

    # Patterns for extracting numerical values with units
    numerical_patterns: dict[str, list[str]] = field(default_factory=lambda: {
        'dimensional': [
            r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*(?:lb|lbs|pounds?|kg|kilos?)',
            r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*(?:"|in|inch|inches|ft|feet)',
            r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*(?:mm|cm|m|meters?)',
        ],
        'performance': [
            r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*(?:hp|kW|mph|rpm)',
            r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*(?:psi|bar|kPa|MPa)',
            r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*(?:gpm|lpm)',
        ]
    })

    # Patterns for identifying descriptive features
    feature_patterns: list[str] = field(default_factory=lambda: [
        r'features (?:a|an) ([^.]+)',
        r'designed (?:for|to) ([^.]+)',
        r'capable of ([^.]+)',
        r'includes (?:a|an) ([^.]+)',
        r'provides ([^.]+capability[^.]+)',
    ])

    # Processing settings
    min_confidence_threshold: float = 0.6
    normalize_units: bool = True

    @property
    def base_metadata(self) -> dict:
        """Provide base metadata structure for all nodes."""
        return {
            "pipeline_name": self.pipeline_name,
            "hierarchy_config": {
                "chunk_sizes": self.hierarchy_config.chunk_sizes,
                "chunk_overlaps": self.hierarchy_config.chunk_overlaps
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