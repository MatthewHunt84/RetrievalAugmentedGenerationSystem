# query_engine_builder.py
import os
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.postprocessor.cohere_rerank import CohereRerank
from rag_package.query_engine import MultimodalQueryEngine, QA_PROMPT
from rag_package.errors import  QueryEngineError, QueryManagerError
from . import rag_config

class QueryEngineBuilder:
    """
    Handles the creation and configuration of multimodal query engines for RAG applications.
    This class manages the complex setup of retrievers, rerankers, and query engines
    while providing flexibility in configuration and robust error checking.
    """

    def __init__(self,
                 index,
                 multimodal_llm: AnthropicMultiModal = None,
                 similarity_top_k: int = 3):
        """
        Initialize the query engine builder with necessary components.

        Args:
            index: The vector store index to use for retrievals
            multimodal_llm: The multimodal language model for processing queries
            similarity_top_k: Number of top similar documents to retrieve
        """
        self.index = index
        self.similarity_top_k = similarity_top_k
        # Store the multimodal LLM or create a default one if not provided
        self.multimodal_llm = multimodal_llm or AnthropicMultiModal(
            model=rag_config.multimodal_model,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    def create_reranker(self) -> CohereRerank:
        """
        Creates and configures a Cohere reranker instance.
        The reranker helps improve retrieval quality by reordering results
        based on their relevance to the query.

        Returns:
            CohereRerank: Configured reranker instance

        Raises:
            QueryEngineError: If reranker creation fails
        """
        try:
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                raise QueryEngineError(
                    "COHERE_API_KEY environment variable not found"
                )

            return CohereRerank(
                api_key=cohere_api_key,
                top_n=self.similarity_top_k,
                model="rerank-multilingual-v3.0"
            )
        except Exception as e:
            raise QueryEngineError(f"Failed to initialize Cohere reranker: {str(e)}")

    def create_retriever(self):
        """
        Creates a retriever from the vector index with specified configuration.
        The retriever is responsible for finding relevant documents from the index.

        Returns:
            The configured retriever instance

        Raises:
            QueryEngineError: If retriever creation fails
        """
        try:
            print("Creating retriever from index...")
            retriever = self.index.as_retriever(
                similarity_top_k=self.similarity_top_k
            )
            print(f"Retriever created successfully: {type(retriever).__name__}")
            return retriever
        except Exception as e:
            raise QueryEngineError(f"Failed to create retriever: {str(e)}")

    def build_engine(self, use_reranker: bool = False) -> MultimodalQueryEngine:
        """
        Builds and configures a complete multimodal query engine.
        This method brings together all components (retriever, reranker, LLM)
        to create a fully functional query engine.

        Args:
            use_reranker: Whether to include the Cohere reranker in postprocessing

        Returns:
            MultimodalQueryEngine: The configured query engine

        Raises:
            QueryEngineError: If engine creation or verification fails
        """
        try:
            # Create base retriever
            retriever = self.create_retriever()

            # Set up postprocessors (reranker if requested)
            postprocessors = []
            if use_reranker:
                print("Initializing Cohere reranker...")
                reranker = self.create_reranker()
                postprocessors.append(reranker)

            print("Initializing MultimodalQueryEngine...")
            engine = MultimodalQueryEngine(
                retriever=retriever,
                qa_prompt=QA_PROMPT,
                multi_modal_llm=self.multimodal_llm,
                node_postprocessors=postprocessors
            )

            # Verify engine components
            self._verify_engine(engine)

            return engine

        except Exception as e:
            raise QueryEngineError(f"Failed to build query engine: {str(e)}")

    def _verify_engine(self, engine: MultimodalQueryEngine) -> None:
        """
        Verifies that all components of the query engine are properly initialized.
        This internal method ensures the engine is ready for use.

        Args:
            engine: The query engine to verify

        Raises:
            QueryEngineError: If any component verification fails
        """
        if not hasattr(engine, '_retriever'):
            raise QueryEngineError("Engine creation failed - retriever not properly initialized")

        print("Engine components verification:")
        print(f"- Retriever: {type(engine._retriever).__name__}")
        print(f"- QA Prompt: {type(engine._qa_prompt).__name__}")
        print(f"- Multimodal LLM: {type(engine._multi_modal_llm).__name__}")


class QueryResponse:
    """
    Represents a formatted response from the query engine. This class structures
    both the answer and its source information in a clear, consistent format.
    """

    def __init__(self, raw_response, query: str):
        """
        Initialize a QueryResponse with the raw engine response and original query.

        Args:
            raw_response: The direct response from the query engine
            query: The original query string that generated this response
        """
        self.raw_response = raw_response
        self.query = query
        self.answer = str(raw_response)
        self.source_nodes = getattr(raw_response, 'source_nodes', [])

    def get_formatted_response(self, include_sources: bool = True) -> str:
        """
        Creates a well-formatted string containing the response and optional
        source information. The format adapts based on whether sources are included.

        Args:
            include_sources: Whether to include source information in the output

        Returns:
            A formatted string containing the response and optional source information
        """
        # Start with the question
        formatted = f"Question: {self.query}\n"
        formatted += "=" * 80 + "\n"

        # Add the main response
        formatted += f"Answer:\n{self.answer}\n"
        formatted += "=" * 80 + "\n"

        # Add source information only if requested and available
        if include_sources and self.source_nodes:
            formatted += "\nSource Information:"
            for idx, source in enumerate(self.source_nodes, 1):
                formatted += f"\n\nSource {idx}:"
                metadata = source.node.metadata
                formatted += f"\nDocument: {metadata['document_name']}"
                formatted += f"\nPage: {metadata['page_num']}"

        return formatted

    def get_sources(self) -> list[dict]:
        """
        Returns a structured list of source information, useful for programmatic
        access to the reference materials used in generating the response.
        """
        return [
            {
                'document': source.node.metadata['document_name'],
                'page': source.node.metadata['page_num']
            }
            for source in self.source_nodes
        ]


class QueryManager:
    """
    Manages the process of sending queries to the engine and handling responses.
    Provides error handling and flexible formatting options for query results.
    """

    def __init__(self, query_engine):
        """
        Initialize the QueryManager with a configured query engine.

        Args:
            query_engine: The multimodal query engine to use for queries
        """
        self.query_engine = query_engine

    def query(self, question: str, get_sources: bool = True) -> QueryResponse:
        """
        Send a query to the engine and get a formatted response. This method
        handles the entire query process, including error handling and formatting.

        Args:
            question: The query string to send to the engine
            get_sources: Whether to include source information in the output

        Returns:
            QueryResponse: A formatted response object containing the answer
                         and optional source information

        Raises:
            QueryManagerError: If the query fails or response formatting fails
        """
        if not question.strip():
            raise QueryManagerError("Query cannot be empty")

        print("\n=== Processing Query ===")
        print(f"Question: {question}")
        print("\nSearching and analyzing documents...")

        try:
            # Send query to engine
            raw_response = self.query_engine.query(question)

            # Create formatted response
            response = QueryResponse(raw_response, question)

            # Print the formatted response with specified source preference
            print(response.get_formatted_response(include_sources=get_sources))

            return response

        except Exception as e:
            raise QueryManagerError(f"Failed to process query: {str(e)}")
