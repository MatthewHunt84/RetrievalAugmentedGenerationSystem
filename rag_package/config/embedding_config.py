from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from functools import lru_cache

@lru_cache()
def get_embed_model() -> OpenAIEmbedding:
    """
    Cached singleton pattern for embedding model access with explicit configuration.
    """
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=None,  # Will use environment variable
        dimensions=3072,  # Explicitly set for text-embedding-3-large
        api_base=None,  # Use default OpenAI API base
    )
    Settings.embed_model = embed_model
    return embed_model