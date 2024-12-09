# rag_config.py
"""
RAG application settings module.
This works similar to a singleton and lets us keep the project settings centralized.
Give private variables an underscore up front: _private_variable
"""
from dataclasses import dataclass, field
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from pydantic import BaseModel
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

# Loading
input_data_folder = "raw_input_data"

# Parsing
@dataclass
class ParserConfig:
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

use_cached_files: bool = True
parsing_config = ParserConfig()

# Node Creation
# Define structured data models for machinery specifications
class DimensionsSpec(BaseModel):
    length: float | None = None
    width: float | None = None
    height: float | None = None
    weight: float | None = None
    units: str | None = None

class EngineSpec(BaseModel):
    manufacturer: str | None = None
    model: str | None = None
    power_output: float | None = None
    power_units: str | None = None
    fuel_type: str | None = None


class HydraulicSpec(BaseModel):
    system_pressure: float | None = None
    flow_rate: float | None = None
    pressure_units: str | None = None
    flow_units: str | None = None


class MachinerySpec(BaseModel):
    model_name: str | None = None
    category: str | None = None
    dimensions: DimensionsSpec | None = None
    engine: EngineSpec | None = None
    hydraulics: HydraulicSpec | None = None
    operating_capacity: float | None = None
    capacity_units: str | None = None


@dataclass
class HierarchyConfig:
    """Configuration for document hierarchy detection and processing."""

    # Patterns for identifying different hierarchy levels
    header_patterns: dict[int, list[str]] = field(default_factory=lambda: {
        # Level 3 (highest) - Main product categories
        3: [
            r'^#{1,2}\s+.+',  # Markdown h1/h2
            r'^[A-Z\s]{5,}$',  # ALL CAPS HEADERS
            r'^(?:PRODUCT|CATEGORY|SERIES)\s*:\s*.+',
        ],
        # Level 2 - Product models and major sections
        2: [
            r'^#{3}\s+.+',  # Markdown h3
            r'^Model\s+[A-Z0-9-]+:',  # Model numbers
            r'^Specifications:',
            r'^Features and Benefits:',
        ],
        # Level 1 - Subsections and specification groups
        1: [
            r'^#{4,}\s+.+',  # Markdown h4+
            r'^(?:Engine|Hydraulic|Electrical|Dimensions|Safety)\s+Specifications:',
            r'^\d+\.\s+[A-Z][^.]+$',  # Numbered sections
        ]
    })

    # Chunk size multipliers for different hierarchy levels
    chunk_size_multipliers: dict[int, float] = field(default_factory=lambda: {
        3: 2.0,  # Larger chunks for high-level sections
        2: 1.0,  # Standard chunk size for mid-level
        1: 0.5  # Smaller chunks for detailed specifications
    })

    # Overlap multipliers for different hierarchy levels
    overlap_multipliers: dict[int, float] = field(default_factory=lambda: {
        3: 2.0,  # More overlap for high-level context
        2: 1.0,  # Standard overlap for mid-level
        1: 0.5  # Less overlap for detailed specs
    })


@dataclass
class KeywordConfig:
    """Configuration for machinery-related keyword detection."""

    capacity_keywords: list[str] = field(default_factory=lambda: [
        "capacity", "can hold", "maximum", "load"
    ])

    weight_keywords: list[str] = field(default_factory=lambda: [
        "weight", "kg", "tons", "pounds", "lbs"
    ])

    dimension_keywords: list[str] = field(default_factory=lambda: [
        "dimensions", "length", "width", "height", "mm", "cm", "meters"
    ])

    power_keywords: list[str] = field(default_factory=lambda: [
        "power", "hp", "kw", "horsepower", "watts"
    ])

    engine_keywords: list[str] = field(default_factory=lambda: [
        "engine", "motor", "diesel", "gasoline", "fuel", "rpm"
    ])

    hydraulic_keywords: list[str] = field(default_factory=lambda: [
        "hydraulic", "pressure", "flow", "psi", "bar"
    ])

    electrical_keywords: list[str] = field(default_factory=lambda: [
        "voltage", "electrical", "battery", "volts", "amps"
    ])

    safety_keywords: list[str] = field(default_factory=lambda: [
        "safety", "warning", "caution", "danger", "protective"
    ])


@dataclass
class NodeCreationConfig:
    """Enhanced configuration for node creation with hierarchy and keyword detection."""

    # Basic configuration
    parsed_results_path: str = 'parsed_results.json'
    image_dir: str = "data_images"
    chunk_size: int = 512
    chunk_overlap: int = 128
    pipeline_name: str = "third_pipeline"
    output_dir: str = "node_outputs"

    # Hierarchy and keyword configurations
    hierarchy_config: HierarchyConfig = field(default_factory=HierarchyConfig)
    keyword_config: KeywordConfig = field(default_factory=KeywordConfig)

    def create_splitter_map(self) -> dict[str, SentenceSplitter]:
        """Create sentence splitters for each hierarchy level."""
        return {
            f"level_{level}": SentenceSplitter(
                chunk_size=int(self.chunk_size * multiplier),
                chunk_overlap=int(self.chunk_overlap * self.hierarchy_config.overlap_multipliers[level])
            )
            for level, multiplier in self.hierarchy_config.chunk_size_multipliers.items()
        }

    @property
    def splitter_ids(self) -> list[str]:
        """Get the list of splitter IDs in hierarchical order."""
        return [f"level_{level}" for level in sorted(
            self.hierarchy_config.chunk_size_multipliers.keys(), reverse=True)]

    @property
    def base_metadata(self) -> dict:
        """Get base metadata for nodes."""
        return {
            "pipeline_name": self.pipeline_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

# Create different pipeline configurations for testing
default_node_config = NodeCreationConfig()

# Example of a configuration with different hierarchy patterns
aggressive_hierarchy_config = NodeCreationConfig(
    pipeline_name="aggressive_hierarchy",
    chunk_size=768,
    chunk_overlap=192,
    hierarchy_config=HierarchyConfig(
        header_patterns={
            3: [r'^[A-Z\s]{4,}$', r'^={3,}\s*$'],  # More aggressive category detection
            2: [r'^-{3,}\s*$', r'^[A-Z][^.]+:'],  # More inclusive section detection
            1: [r'^\s*[•\-]\s', r'^\d+\.\s']  # Include bullet points and numbered items
        },
        chunk_size_multipliers={3: 3.0, 2: 1.5, 1: 0.75},  # More extreme size differences
        overlap_multipliers={3: 2.5, 2: 1.5, 1: 0.75}
    )
)

# Example of a configuration focusing on technical specifications
technical_spec_config = NodeCreationConfig(
    pipeline_name="technical_focus",
    chunk_size=384,
    chunk_overlap=96,
    keyword_config=KeywordConfig(
        # Enhanced technical keyword lists
        engine_keywords=["engine", "motor", "diesel", "gasoline", "fuel", "rpm", "displacement", "cylinders"],
        hydraulic_keywords=["hydraulic", "pressure", "flow", "psi", "bar", "pump", "valve", "circuit"]
    )
)

# Make the default configuration available
node_config = default_node_config

# Vector store
embedding_model_name = "text-embedding-3-large"

# Query Engine
multimodal_model = "claude-3-5-sonnet-latest"
multimodal_llm = AnthropicMultiModal(model=multimodal_model)

def get_embed_model():
    if not hasattr(get_embed_model, '_instance'):
        get_embed_model._instance = OpenAIEmbedding(model="text-embedding-3-large")
        # Update LlamaIndex settings
        Settings.embed_model = get_embed_model._instance
    return get_embed_model._instance