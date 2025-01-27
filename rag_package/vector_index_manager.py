# vector_index_manager.py
import json
import logging
from pathlib import Path
from rag_package.errors import VectorIndexError
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core import VectorStoreIndex, StorageContext, Settings, ServiceContext
from pathlib import Path
from typing import List
from llama_index.core.schema import TextNode, BaseNode
import copy


class VectorIndexManager:
    """
    Manages vector index operations including creation, storage, and loading.
    This class uses LlamaIndex's new Settings system for configuration and maintains
    its own logger for tracking operations and debugging.
    """

    def __init__(
            self,
            embed_model,
            storage_dir: str = "./storage_manuals"
    ):
        """
        Initialize the VectorIndexManager with storage location and embedding configuration.
        We set up logging first, then configure the embedding model and storage location.

        Args:
            embed_model: The embedding model to use for vector operations
            storage_dir: Directory where the vector index will be stored
        """
        # Set up logging first - this needs to happen before any other operations
        self.logger = logging.getLogger(__name__)

        # Basic setup
        self.storage_path = Path(storage_dir)
        self.embed_model = embed_model

        # Configure global settings for LlamaIndex
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512  # Size of text chunks for processing
        Settings.chunk_overlap = 20  # Amount of overlap between chunks

        # Log initialization details
        self.logger.info(f"Initialized VectorIndexManager with storage at: {self.storage_path}")
        self.logger.info(f"Using embedding model: {self.embed_model.__class__.__name__}")

    def index_exists(self) -> bool:
        """
        Check if a vector index already exists in the storage directory.

        Returns:
            True if an index exists, False otherwise
        """
        return self.storage_path.exists()

    def _prepare_nodes_for_indexing(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """
        Prepares nodes for indexing by ensuring all metadata and relationships
        follow LlamaIndex's expected structure. This is particularly important for
        maintaining hierarchical relationships between nodes while ensuring proper
        serialization.

        Args:
            nodes: List of BaseNode objects to prepare for indexing

        Returns:
            List[BaseNode]: Nodes with properly structured, serializable data
        """
        from llama_index.core.schema import RelatedNodeInfo, NodeRelationship

        prepared_nodes = []
        for node in nodes:
            # Create a deep copy to avoid modifying the original
            node_copy = copy.deepcopy(node)

            # Clean the metadata dictionary
            cleaned_metadata = {}
            for key, value in node_copy.metadata.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    cleaned_value = {}
                    for k, v in value.items():
                        if isinstance(v, set):
                            cleaned_value[k] = list(v)
                        else:
                            cleaned_value[k] = v
                    cleaned_metadata[key] = cleaned_value
                else:
                    cleaned_metadata[key] = value

            # Replace metadata with cleaned version
            node_copy.metadata = cleaned_metadata

            # Handle relationships using LlamaIndex's proper structures
            if hasattr(node_copy, 'relationships'):
                new_relationships = {}
                for rel_type, rel_ids in node_copy.relationships.items():
                    # Convert relationship type to proper enum
                    if rel_type == "child":
                        enum_type = NodeRelationship.CHILD
                    elif rel_type == "parent":
                        enum_type = NodeRelationship.PARENT
                    else:
                        continue  # Skip unknown relationship types

                    # Convert IDs to RelatedNodeInfo objects
                    related_nodes = []
                    for node_id in rel_ids:
                        # Create proper RelatedNodeInfo object for each related node
                        related_nodes.append(
                            RelatedNodeInfo(
                                node_id=node_id,
                                node_type=None,  # Optional, can be None
                                metadata={}  # Optional metadata about the relationship
                            )
                        )

                    new_relationships[enum_type] = related_nodes

                # Replace relationships with properly structured version
                node_copy.relationships = new_relationships

            # Clear any cached embeddings to prevent serialization issues
            if hasattr(node_copy, 'embedding'):
                node_copy.embedding = None

            prepared_nodes.append(node_copy)

        self.logger.info(f"Prepared {len(prepared_nodes)} nodes for indexing")
        if prepared_nodes and self.logger.level <= logging.DEBUG:
            # Log a sample of the prepared structure for debugging
            sample_node = prepared_nodes[0]
            self.logger.debug(
                f"Sample node structure:\n"
                f"Metadata: {json.dumps(sample_node.metadata, indent=2)}\n"
                f"Relationships: {sample_node.relationships}"
            )

        return prepared_nodes

    def create_index(self, text_nodes: list[BaseNode]) -> VectorStoreIndex:
        """
        Creates a new vector index from text nodes with comprehensive preprocessing.

        Args:
            text_nodes: List of BaseNode objects to index

        Returns:
            VectorStoreIndex: The created and saved index
        """
        try:
            self.logger.info(f"Creating new vector index from {len(text_nodes)} text nodes...")

            # Prepare nodes with enhanced cleaning
            prepared_nodes = self._prepare_nodes_for_indexing(text_nodes)

            # Create index with explicit settings
            Settings.embed_model = self.embed_model
            Settings.chunk_size = 512
            Settings.chunk_overlap = 20

            index = VectorStoreIndex(
                nodes=prepared_nodes,
                show_progress=True,
                storage_context=StorageContext.from_defaults()
            )

            self.logger.info(f"Successfully created index with {len(prepared_nodes)} nodes")

            # Save the index
            self.logger.info(f"Saving index to {self.storage_path}...")
            index.storage_context.persist(persist_dir=str(self.storage_path))

            return index

        except Exception as e:
            self.logger.error(f"Failed to create index: {str(e)}")
            import traceback
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise VectorIndexError(f"Failed to create index: {str(e)}")

    def load_index(self) -> VectorStoreIndex:
        """
        Loads an existing vector index using the new Settings approach.
        """
        try:
            self.logger.info("Loading existing index from storage...")

            # Configure settings
            Settings.embed_model = self.embed_model

            # Load the storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.storage_path)
            )

            # Load the index with the new approach
            index = load_index_from_storage(
                storage_context=storage_context
            )

            if not hasattr(index, 'index_struct'):
                raise VectorIndexError("Loaded index is missing required structure")

            # Verify the index is working
            num_docs = len(index.docstore.docs) if hasattr(index, 'docstore') else 0
            self.logger.info(f"Successfully loaded index with {num_docs} documents")

            return index

        except Exception as e:
            self.logger.error(f"Failed to load index: {str(e)}")
            import traceback
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise VectorIndexError(f"Failed to load index: {str(e)}")

    def get_or_create_index(self, text_nodes: List[TextNode] = None) -> VectorStoreIndex:
        """
        Gets existing index or creates a new one with better error handling.
        """
        try:
            if self.index_exists():
                try:
                    # Test embedding model
                    test_embedding = self.embed_model.get_query_embedding("test query")
                    print(f"Embedding model test successful - generated embedding of size {len(test_embedding)}")
                    return self.load_index()
                except Exception as e:
                    print(f"Error testing embedding model: {str(e)}")
                    raise

            if text_nodes is None:
                raise VectorIndexError("Cannot create new index: text_nodes not provided")

            # Verify nodes before creating index
            print(f"Verifying {len(text_nodes)} nodes...")
            for i, node in enumerate(text_nodes[:3]):  # Check first 3 nodes
                print(f"Node {i} type: {type(node)}")
                print(f"Node {i} text length: {len(node.text)}")
                print(f"Node {i} metadata keys: {list(node.metadata.keys())}")

            return self.create_index(text_nodes)

        except Exception as e:
            import traceback
            print("Full error traceback:")
            traceback.print_exc()
            raise VectorIndexError(f"Failed to create or load vector index: {str(e)}")
