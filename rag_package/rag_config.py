#rag_config.py
"""
RAG application settings module.
This works similar to a singleton and lets us keep the project settings centralized.
Give private variables an underscore up front: _private_variable
"""

import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from dotenv import load_dotenv

# Public variables
parser_result_type = "markdown"
embedding_model_name = "text-embedding-3-large"

# LlamaIndex settings

# embed_model = OpenAIEmbedding(model="text-embedding-3-large")
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# Generative language model
multimodal_model_name = "claude-3-5-sonnet-latest"
multimodal_llm = AnthropicMultiModal(model="claude-3-5-sonnet-latest")

# embed_model = OpenAIEmbedding(
#     model="text-embedding-3-large",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     timeout=10,  # Increase timeout
#     max_retries=3  # Add retries
# )

def get_embed_model():
    if not hasattr(get_embed_model, '_instance'):
        get_embed_model._instance = OpenAIEmbedding(model="text-embedding-3-large")
        # Update LlamaIndex settings
        Settings.embed_model = get_embed_model._instance
    return get_embed_model._instance