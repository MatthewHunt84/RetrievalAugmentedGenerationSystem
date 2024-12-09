# rag_config.py
"""
RAG application settings module.
This works similar to a singleton and lets us keep the project settings centralized.
Give private variables an underscore up front: _private_variable
"""
from dataclasses import dataclass
from typing import List
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from pydantic import BaseModel
from llama_index.core import Settings
from llama_index.core.extractors import (
    KeywordExtractor,
    PydanticProgramExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)

# Loading
input_data_folder = "raw_input_data"

# Parsing
@dataclass
class ParserConfig:
    result_type: str = "markdown"
    parsing_instruction: str = "You are given product line brochure for heavy machinery"
    use_vendor_multimodal_model: bool = True
    vendor_multimodal_model_name: str = "anthropic-sonnet-3.5"
    show_progress: bool = True
    verbose: bool = True
    invalidate_cache: bool = True
    do_not_cache: bool = False
    num_workers: int = 8
    language: str = "en"
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
class NodeCreationConfig:
    parsed_results_path: str = 'parsed_results.json'
    image_dir: str = "data_images"
    chunk_size: int = 512
    chunk_overlap: int = 128
    pipeline_name: str = "first_pipeline"
    output_dir: str = "node_outputs"

    @property
    def metadata_extractors(self) -> List:
        return [
            TitleExtractor(nodes=5),
            KeywordExtractor(
                keywords=["capacity", "weight", "dimensions", "power",
                          "engine", "hydraulic", "electrical", "safety"]
            ),
            QuestionsAnsweredExtractor(),
            SummaryExtractor(summaries=["brief", "detailed"]),
            PydanticProgramExtractor(
                programs=[MachinerySpec],
                extract_chunks_with_metadata=True
            )
        ]

    @property
    def base_metadata(self) -> dict:
        return {
            "pipeline_name": self.pipeline_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
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