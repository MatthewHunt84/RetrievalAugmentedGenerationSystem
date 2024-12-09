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

from dataclasses import dataclass, field
from llama_index.core.node_parser import SentenceSplitter


@dataclass
class TechnicalSpecPatterns:
    """Patterns for extracting technical specifications from machinery documentation."""
    dimensional_patterns: dict[str, list[str]] = field(default_factory=lambda: {
        'weight': [
            r'(\d+(?:\.\d+)?)\s*(?:lb|lbs|pounds?|kg|kilos?)',
            r'weighs?\s+(\d+(?:\.\d+)?)\s*(?:lb|lbs|pounds?|kg|kilos?)'
        ],
        'working_depth': [
            r'(?:dig|trench|working) depth(?:s)? (?:of )?(\d+(?:-\d+)?)\s*(?:"|inches|ft|feet)',
            r'depth(?:s)? (?:of )?(\d+(?:-\d+)?)\s*(?:"|inches|ft|feet)'
        ],
        'pressure': [
            r'(\d+(?:\.\d+)?)\s*(?:psi|bar|kPa)',
            r'pressure of (\d+(?:\.\d+)?)\s*(?:psi|bar|kPa)'
        ]
    })

    performance_patterns: dict[str, list[str]] = field(default_factory=lambda: {
        'power': [
            r'(\d+(?:\.\d+)?)\s*(?:hp|horsepower|kW)',
            r'power output of (\d+(?:\.\d+)?)\s*(?:hp|horsepower|kW)'
        ],
        'flow_rate': [
            r'(\d+(?:\.\d+)?)\s*(?:gpm|lpm)',
            r'flow(?:s)? at (\d+(?:\.\d+)?)\s*(?:gpm|lpm)'
        ]
    })


@dataclass
class FeatureExtractionPatterns:
    """Patterns for identifying key features and capabilities."""
    control_features: list[str] = field(default_factory=lambda: [
        r'features (?:a|an) ([^.]+control[^.]+)',
        r'includes (?:a|an) ([^.]+system[^.]+)'
    ])

    applications: list[str] = field(default_factory=lambda: [
        r'designed for ([^.]+)',
        r'can (?:easily )?handle ([^.]+)',
        r'suitable for ([^.]+)'
    ])

    maintenance_features: list[str] = field(default_factory=lambda: [
        r'reduces maintenance ([^.]+)',
        r'easy to maintain ([^.]+)',
        r'simplified (?:maintenance|service) ([^.]+)'
    ])


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical document processing."""
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

    # Chunk sizes for different hierarchy levels
    chunk_sizes: dict[int, int] = field(default_factory=lambda: {
        3: 1024,  # Larger chunks for main sections
        2: 512,  # Medium chunks for model details
        1: 256  # Smaller chunks for specifications
    })

    # Overlap sizes for different hierarchy levels
    chunk_overlaps: dict[int, int] = field(default_factory=lambda: {
        3: 256,  # 25% overlap for main sections
        2: 128,  # 25% overlap for model details
        1: 64  # 25% overlap for specifications
    })


@dataclass
class NodeCreationConfig:
    """
    Comprehensive configuration combining hierarchical parsing with technical
    metadata extraction capabilities.
    """
    # Pipeline identification
    pipeline_name: str = 'fourth_pipeline'

    # File paths
    parsed_results_path: str = 'parsed_results.json'
    image_dir: str = "data_images"
    output_dir: str = "node_outputs"

    # Processing configurations
    hierarchy_config: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    technical_patterns: TechnicalSpecPatterns = field(default_factory=TechnicalSpecPatterns)
    feature_patterns: FeatureExtractionPatterns = field(default_factory=FeatureExtractionPatterns)

    # Processing settings
    normalize_units: bool = True
    confidence_threshold: float = 0.8

    @property
    def base_metadata(self) -> dict:
        """Get base metadata for nodes."""
        return {
            "pipeline_name": self.pipeline_name,
            "hierarchy_config": {
                "chunk_sizes": self.hierarchy_config.chunk_sizes,
                "chunk_overlaps": self.hierarchy_config.chunk_overlaps
            }
        }

node_config = NodeCreationConfig()

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