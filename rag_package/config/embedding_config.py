from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

def get_embed_model() -> OpenAIEmbedding:
    """
    Singleton pattern for embedding model access.
    Ensures consistent embedding model usage across the application.
    """
    if not hasattr(get_embed_model, '_instance'):
        get_embed_model._instance = OpenAIEmbedding(model="text-embedding-3-large")
        Settings.embed_model = get_embed_model._instance
    return get_embed_model._instance