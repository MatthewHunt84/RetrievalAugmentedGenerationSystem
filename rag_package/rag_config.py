#rag_config.py
"""
RAG application settings module.
This works similar to a singleton and lets us keep the project settings centralized.
Give private variables an underscore up front: _private_variable
"""
from dataclasses import dataclass
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.core import Settings

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

# Chunking

# Vector store

embedding_model_name = "text-embedding-3-large"

# LLM querying

multimodal_model = "claude-3-5-sonnet-latest"
multimodal_llm = AnthropicMultiModal(model=multimodal_model)

def get_embed_model():
    if not hasattr(get_embed_model, '_instance'):
        get_embed_model._instance = OpenAIEmbedding(model="text-embedding-3-large")
        # Update LlamaIndex settings
        Settings.embed_model = get_embed_model._instance
    return get_embed_model._instance

