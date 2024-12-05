# main.py
import os
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

    # Settings.embed_model = rag_config.embed_model

    try:
        # Initialize our managers
        document_processor = DocumentProcessor()
        node_creator = TextNodeCreator()
        index_manager = VectorIndexManager(embed_model=embed_model)

        # If we need to create a new index
        if not index_manager.index_exists():
            # Process documents and create text nodes
            document_processor.process_all(extract_images=True)
            text_nodes = node_creator.create_nodes()
            # Create new index
            index = index_manager.get_or_create_index(text_nodes, multimodal_model=rag_config.multimodal_llm)
        else:
            # Just load existing index
            index = index_manager.get_or_create_index()

        # Create query engine
        print("Creating query engine...")  # Add debug print
        engine_builder = QueryEngineBuilder(index)
        query_engine = engine_builder.build_engine(use_reranker=False)

        # Initialize query manager
        print("Initializing query manager...")  # Add debug print
        query_manager = QueryManager(query_engine)

        # Send test query without sources
        test_query = "What parts are included in the Uppspel?"
        print(f"Sending test query: {test_query}")  # Add debug print
        response = query_manager.query(test_query, get_sources=False)

    except Exception as e:
        print(f"Error type: {type(e)}")  # Print error type
        print(f"Error details: {str(e)}")  # Print error details
        return error_handler.handle_error(e)

    return 0


# Main execution block - This is where we orchestrate everything
if __name__ == "__main__":
    main()