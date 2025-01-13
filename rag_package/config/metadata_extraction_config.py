"""
Configuration module for metadata extraction in the RAG pipeline.

This module defines the MetadataExtractionConfig class which manages settings
for extracting metadata from technical documentation, supporting multiple LLM providers.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal, TypeAlias
from rag_package.models.llm_clients import BaseLLMClient, create_llm_client

# Updated type definitions to include OpenAI models
ModelChoice: TypeAlias = Literal["haiku", "sonnet", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct"]

@dataclass
class MetadataExtractionConfig:
    """
    Configuration settings for metadata extraction process.

    Required Parameters:
        model: The model choice for metadata extraction
              ("haiku", "sonnet", "gpt-4", "gpt-3.5-turbo")

    Optional Parameters:
        batch_size: Number of model descriptions to process in a batch
        temperature: Temperature setting for the model
        confidence_threshold: Threshold for metadata matching confidence

    Example:
        metadata_config = MetadataExtractionConfig(
            model="gpt-3.5-turbo"
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
    _llm_client: Optional[BaseLLMClient] = None
    model_name: str = field(init=False)

    def __post_init__(self):
        """Initialize the LLM client based on the model choice."""
        self._llm_client = create_llm_client(self.model, self.temperature)
        self.model_name = self._llm_client.model_name

    def get_client(self) -> BaseLLMClient:
        """Get the LLM client instance."""
        if self._llm_client is None:
            self._llm_client = create_llm_client(self.model, self.temperature)
        return self._llm_client

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
