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
from anthropic import Anthropic, Client  # Add this import
from llama_index.core import Settings
from pydantic import BaseModel, Field
import re

# Base directory for raw input data files


# Global configuration instances
input_data_folder: str = "raw_input_data"
embedding_model_name = "text-embedding-3-large"
# multimodal_model = "claude-3-5-sonnet-latest"
multimodal_model = "claude-3-5-haiku-latest"
multimodal_llm = AnthropicMultiModal(model=multimodal_model)
metadata_extraction_model = "claude-3-5-sonnet-latest"

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

"""
Metadata Extraction: Pydantic models for equipment metadata
"""
class EquipmentMetadata(BaseModel):
    """Metadata schema for equipment product information."""
    product_name: str = Field(..., description="The full name of the product")
    model_number: str = Field(..., description="The specific model number/identifier of the product")
    manufacturer: str = Field(..., description="The manufacturer of the equipment")
    category: str = Field(..., description="Main equipment category")
    subcategory: str | None = Field(None, description="More specific classification within the main category")
    year: str | None = Field(None, description="Manufacturing or model year if available")
    document_type: str = Field(..., description="Type of document (e.g., 'catalog', 'manual', 'spec sheet')")
    content_types: list[str] = Field(default_factory=list, description="Types of content present for this model")


@dataclass
class MetadataExtractionConfig:
    """Configuration settings for metadata extraction process."""

    # Batch size for processing model descriptions
    batch_size: int = 5

    # LLM model configuration
    model_name: str = metadata_extraction_model
    temperature: float = 0.1

    # Store the actual LLM instance
    _llm_instance: [Client] = None

    def get_client(self) -> Client:
        """Get or create the Anthropic client instance."""
        if self._llm_instance is None:
            self._llm_instance = Anthropic()
        return self._llm_instance

    # Updated extraction prompts
    document_level_prompt: str = """You are analyzing a technical equipment catalog. 
    Extract the following document-level information in JSON format:
    - Equipment categories present in the document
    - Manufacturer information
    - Document type (e.g., catalog, manual, spec sheet)
    - Publication year if available

    Categorize equipment naturally based on industry standards and the document's content.

    Text to analyze: {text}
    """

    model_batch_prompt: str = """You are extracting metadata for equipment models.
    For each distinct model in the provided text, extract in JSON format:
    - Full product name
    - Model number
    - Manufacturer
    - Category (determine based on equipment type and industry standards)
    - Subcategory (more specific classification if applicable)
    - Year (if mentioned)
    - Content types (types of information present in the description)

    Use natural, industry-standard categorizations. Be specific but consistent. 
    If some of these properties are not mentioned please return an empty string as the value.
    Identify any content types that seem relevant (e.g., specifications, features, applications, maintenance).

    Text to analyze: {text}
    """

    # Confidence threshold for metadata matching
    confidence_threshold: float = 0.8

@dataclass
class NodeCreationConfig:
    """Configuration for node creation."""
    pipeline_name: str = 'eighth_pipeline'
    parsed_results_path: str = 'parsed_results.json'
    output_dir: str = "node_outputs"

    chunk_sizes: list[int] = field(default_factory=lambda: [2048, 512, 128])
    chunk_overlap: int = 20

    # Keep the hierarchy config
    hierarchy_config: HierarchicalConfig = field(default_factory=HierarchicalConfig)

    # Metadata extraction config
    metadata_extraction: MetadataExtractionConfig = field(default_factory=MetadataExtractionConfig)

    @property
    def base_metadata(self) -> dict:
        return {
            "pipeline_info": {
                "name": self.pipeline_name
            },
            "extraction_info": {
                "metadata_version": "1.0", # Metadata extraction info
                "extraction_timestamp": None  # Will be set during processing
            }
        }

node_config = NodeCreationConfig()

def get_embed_model() -> OpenAIEmbedding:
    """
    Singleton pattern for embedding model access.
    Ensures consistent embedding model usage across the application.
    """
    if not hasattr(get_embed_model, '_instance'):
        get_embed_model._instance = OpenAIEmbedding(model="text-embedding-3-large")
        Settings.embed_model = get_embed_model._instance
    return get_embed_model._instance