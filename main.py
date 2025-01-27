# main.py
from dotenv import load_dotenv
from torch.utils.model_dump import hierarchical_pickle

from rag_package.config.hierarchical_config import HierarchicalConfig
from rag_package.config.metadata_extraction_config import MetadataExtractionConfig
from rag_package.document_processor import DocumentProcessor
from rag_package.vector_index_manager import VectorIndexManager
from rag_package.textnode_creator import TextNodeCreator
from rag_package.query_engine_builder import QueryEngineBuilder
from rag_package.query_engine_builder import QueryManager
from rag_package.error_handler import ErrorHandler
from rag_package.config.parser_config import ParserConfig
from rag_package.config.node_creation_config import NodeCreationConfig
from rag_package.config.embedding_config import get_embed_model
from rag_package import structured_query_engine
import llama_index

def main():
    load_dotenv()
    error_handler = ErrorHandler()

    try:
        # If we need to parse files and create nodes
        # if not index_manager.index_exists():
        #     print("index does not exist")
        # else:
        #     print("index exists")

        # print(f"Embedding model type: {type(embed_model)}")
        # print(f"Embedding model class: {embed_model.__class__.__name__}")
        #
        # # Test the embedding model
        # try:
        #     test_embedding = embed_model.get_query_embedding("test")
        #     print(f"Successfully generated test embedding of length: {len(test_embedding)}")
        # except Exception as e:
        #     print(f"Error testing embedding model: {str(e)}")

        # parser = ParserConfig(model="sonnet_multimodal", use_cached_files=True)
        #
        # document_processor = DocumentProcessor(parsing_config=parser)
        # document_processor.process_all()
        #
        node_creation_config = NodeCreationConfig(pipeline_name="seventeenth_pipeline_gpt_instruct",
                                                  hierarchical_config=HierarchicalConfig(),
                                                  metadata_extraction=MetadataExtractionConfig(model="gpt-3.5-turbo-instruct"),
                                                  test_pages=[10])

        node_creator = TextNodeCreator(node_config=node_creation_config)

        text_nodes = node_creator.create_or_load_nodes()

        # embed_model = get_embed_model()

        # index_manager = VectorIndexManager(embed_model=embed_model)
        # index = index_manager.get_or_create_index(text_nodes)

        # structured_query_engine.query(index)





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