"""
Configuration module for metadata extraction in the RAG pipeline.

This module defines the MetadataExtractionConfig class which manages settings
for extracting metadata from technical documentation.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal, TypeAlias

from anthropic import Client

# Type definitions for metadata extraction configuration
ModelChoice: TypeAlias = Literal["haiku", "sonnet"]


@dataclass
class MetadataExtractionConfig:
    """
    Configuration settings for metadata extraction process.

    Required Parameters:
        model: The model choice for metadata extraction ("haiku", "sonnet", "opus")

    Optional Parameters:
        batch_size: Number of model descriptions to process in a batch
        temperature: Temperature setting for the model
        confidence_threshold: Threshold for metadata matching confidence

    Example:
        metadata_config = MetadataExtractionConfig(
            model="haiku"
        )
    """
    # Required parameters
    model: ModelChoice

    # Optional parameters
    batch_size: int = 5
    temperature: float = 0.1
    confidence_threshold: float = 0.8

    # Token management
    max_tokens_per_batch: int = 12000
    tokens_for_prompt: int = 500
    tokens_for_response: int = 2000
    metadata_matching_threshold: float = 0.1

    # Private fields
    _llm_instance: Optional[Client] = None
    model_name: str = field(init=False)

    def __post_init__(self):
        """Set the full model name based on the model choice."""
        self.model_name = MODEL_VENDOR_MAPPING[self.model]

    def get_client(self) -> Client:
        """Get or create the Anthropic client instance."""
        if self._llm_instance is None:
            self._llm_instance = Client()
        return self._llm_instance

    # Extraction prompts
    document_level_prompt: str = '''Analyze this text and extract document-level metadata.
        Focus on identifying overall document characteristics.

        Provide ONLY a JSON object with this exact structure:
        {{
            "manufacturer": "primary manufacturer name",
            "document_type": "type of document (e.g., catalog, brochure, spec sheet)",
            "year_published": "publication or latest year mentioned",
            "equipment_categories": ["list of equipment categories found"],
            "models_included": ["list of all model numbers detailed in the file"]
        }}

        Text to analyze: {text}
        '''

    model_batch_prompt: str = '''Analyze each equipment model description and extract metadata in JSON format.
        For each distinct model provide:
        {{
            "product_name": "full product name",
            "model_number": "specific model identifier",
            "manufacturer": "manufacturer name",
            "category": "equipment category",
            "subcategory": "more specific classification if applicable",
            "year": "manufacturing/model year if mentioned",
            "specifications": ["specifications for this model such as size, weight"],
            "capabilities": ["capabilities or features that differentiate this model"],
            "content_types": ["list of content types present in description"]
        }}

        Text to analyze: {text}

        Remember: Respond ONLY with the JSON array containing metadata for each model found.
        '''


MODEL_VENDOR_MAPPING = {
    "haiku": "claude-3-5-haiku-latest",
    "sonnet": "claude-3-5-sonnet-latest",
}