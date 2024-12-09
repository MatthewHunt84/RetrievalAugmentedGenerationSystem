# main.py
from dotenv import load_dotenv
from rag_package.document_processor import DocumentProcessor
from rag_package.vector_index_manager import VectorIndexManager
from rag_package.textnode_creator import TextNodeCreator
from rag_package.query_engine_builder import QueryEngineBuilder
from rag_package.query_engine_builder import QueryManager
from rag_package.error_handler import ErrorHandler
from rag_package import rag_config


def main():
    load_dotenv()
    error_handler = ErrorHandler()
    embed_model = rag_config.get_embed_model()

    try:
        # index_manager = VectorIndexManager(embed_model=embed_model)

        # If we need to parse files and create nodes
        # if not index_manager.index_exists():
        # document_processor = DocumentProcessor(parsing_config=rag_config.parsing_config)
        # document_processor.process_all(use_cached_files=rag_config.use_cached_files)
        node_creator = TextNodeCreator(node_config=rag_config.node_config)
        nodes = node_creator.create_nodes()
    #         # Create new index
    #         index = index_manager.get_or_create_index(text_nodes)
    #     else:
    #         # Otherwise use existing vectors for query
    #         index = index_manager.get_or_create_index()
    #
    #     # Create query engine
    #     engine_builder = QueryEngineBuilder(index)
    #     query_engine = engine_builder.build_engine(use_reranker=False)
    #     query_manager = QueryManager(query_engine)
    #
    #     # Submit query to llm
    #     test_query = "What parts are included in the Uppspel?"
    #     print(f"Sending test query: {test_query}")  # Add debug print
    #     response = query_manager.query(test_query, get_sources=True)
    #


    except Exception as e:
        return error_handler.handle_error(e)

    return 0


# Main execution block - This is where we orchestrate everything
if __name__ == "__main__":
    main()