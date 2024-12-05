# vector_index_manager.py
from pathlib import Path
from rag_package.errors import VectorIndexError
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)

class VectorIndexManager:
    """
    Manages vector index operations including creation, storage, and loading.
    This class handles the lifecycle of vector indices, providing a seamless interface
    for working with either new or existing indices while managing the expensive
    computational resources involved.
    """

    def __init__(
            self,
            embed_model,
            storage_dir: str = "./storage_manuals"
    ):
        """
        Initialize the VectorIndexManager with storage location and embedding model.

        Args:
            embed_model: Embedding model that provides embed_query and embed_documents methods
            storage_dir: Directory where the vector index will be stored

        Raises:
            VectorIndexError: If the embedding model doesn't meet the required criteria
        """
        self.storage_path = Path(storage_dir)
        self.embed_model = embed_model

    def index_exists(self) -> bool:
        """
        Check if a vector index already exists in the storage directory.

        Returns:
            True if an index exists, False otherwise
        """
        return self.storage_path.exists()

    def create_index(self, text_nodes: list) -> 'VectorStoreIndex':
        """
        Creates a new vector index from text nodes and saves it to storage.
        This is an expensive operation that makes API calls to create embeddings.

        Args:
            text_nodes: List of TextNode objects to create the index from

        Returns:
            VectorStoreIndex: The newly created and saved index

        Raises:
            VectorIndexError: If index creation fails
        """
        try:
            print("Creating new vector index from text nodes...")
            index = VectorStoreIndex(
                text_nodes,
                embed_model=self.embed_model
            )

            print(f"Saving index to {self.storage_path} for future use...")
            index.storage_context.persist(persist_dir=str(self.storage_path))

            return index

        except Exception as e:
            raise VectorIndexError(f"Failed to create index: {str(e)}")

    def load_index(self) -> 'VectorStoreIndex':
        """
        Loads an existing vector index from storage and verifies its functionality.
        This is a cheaper operation as it doesn't require API calls for embedding creation.

        Returns:
            VectorStoreIndex: The loaded and verified index

        Raises:
            VectorIndexError: If loading or verification fails
        """
        try:
            print("Loading existing index from storage...")

            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.storage_path)
            )

            index = load_index_from_storage(
                storage_context=storage_context,
                embed_model=self.embed_model
            )

            if not hasattr(index, 'index_struct'):
                raise VectorIndexError("Loaded index is missing required structure")

            num_docs = len(index.docstore.docs) if hasattr(index, 'docstore') else 0

            print("Successfully loaded index with the following properties:")
            print(f"- Number of documents in docstore: {num_docs}")
            print(f"- Using embedding model: {self.embed_model.__class__.__name__}")

            try:
                retriever = index.as_retriever()
                print("- Successfully verified retriever creation")
            except Exception as e:
                raise VectorIndexError(f"Failed to create retriever: {str(e)}")

            return index

        except Exception as e:
            raise VectorIndexError(f"Failed to load index: {str(e)}")

    def get_or_create_index(self, text_nodes: list = None) -> 'VectorStoreIndex':
        try:
            if self.index_exists():
                print("Testing embedding model before loading index...")
                # Test the embedding model
                try:
                    _ = self.embed_model.get_query_embedding("test")
                    print("Embedding model test successful")
                except Exception as e:
                    print(f"Embedding model test failed: {str(e)}")
                    raise

                return self.load_index()
            else:
                if text_nodes is None:
                    raise VectorIndexError(
                        "Cannot create new index: text_nodes not provided"
                    )
                return self.create_index(text_nodes)

        except VectorIndexError as e:
            if text_nodes is not None:
                print(f"Error loading index: {str(e)}")
                print("Attempting to create new index instead...")
                return self.create_index(text_nodes)
            raise
