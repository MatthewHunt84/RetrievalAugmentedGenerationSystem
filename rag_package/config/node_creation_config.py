"""
Configuration module for node creation in the RAG pipeline.

This module defines the NodeCreationConfig class which manages configuration
for creating nodes from technical documentation, including hierarchical structure
and metadata extraction settings.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Literal, TypeAlias
from pathlib import Path

from .hierarchical_config import HierarchicalConfig
from .metadata_extraction_config import MetadataExtractionConfig
from .logging_config import LoggingConfig
from .path_config import PathConfig

@dataclass
class NodeCreationConfig:
    """
    Configuration for the node creation process in the RAG pipeline.

    Required Parameters:
        pipeline_name: Name identifier for the pipeline
        hierarchical_config: Configuration for managing document hierarchy
        metadata_extraction_config: Configuration for metadata extraction

    Example:
        node_config = NodeCreationConfig(
            pipeline_name="equipment_catalog_pipeline",
            hierarchical_config=hierarchical_config,
            metadata_extraction_config=metadata_config
        )
    """

    # Required dependencies
    pipeline_name: str
    hierarchical_config: HierarchicalConfig
    metadata_extraction: MetadataExtractionConfig

    # Processing configuration
    use_cached_files: bool = True
    invalidate_cache: bool = False
    do_not_cache: bool = False

    # Optional configurations
    paths_config: PathConfig = field(default_factory=PathConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)

    # Processing behavior
    show_progress: bool = True
    verbose: bool = True
    num_workers: int = 8

    # Test configuration
    test_pages: Optional[List[int]] = None

    # Chunking configuration
    chunk_sizes: List[int] = field(default_factory=lambda: [2048, 512, 128])
    chunk_overlap: int = 20

    # Base instruction that applies to all processing
    base_instruction: str = """
    Process technical documentation maintaining:
    1. Document hierarchy through header levels
    2. Technical specifications and measurements
    3. Model information and identifiers
    4. Detailed feature descriptions
    5. Safety information
    6. Cross-references and relationships
    7. Technical terminology

    Maintain structural integrity:
    - Preserve header hierarchy
    - Keep related content together
    - Maintain technical accuracy
    - Preserve formatting where significant
    - Keep all numerical values exact
    """

    @property
    def base_metadata(self) -> dict:
        """
        Generates base metadata dictionary for the node creation process.

        Returns:
            dict: Base metadata including pipeline information and extraction details
        """
        return {
            "pipeline_info": {
                "name": self.pipeline_name,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extraction_info": {
                "extraction_model": self.metadata_extraction.model_name,
                "metadata_version": "1.0",
                "extraction_timestamp": datetime.now().isoformat()
            }
        }