# main.py
from dotenv import load_dotenv
from torch.utils.model_dump import hierarchical_pickle

from rag_package.config.hierarchical_config import HierarchicalConfig
from rag_package.config.metadata_extraction_config import MetadataExtractionConfig
from rag_package.csv_creator import CSVCreator
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
from rag_package import csv_creator

from rag_package.structured_query_engine import StructuredQueryEngineBuilder
import llama_index
import json

import asyncio

async def main():
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
        # node_creation_config = NodeCreationConfig(pipeline_name="eighteenth_pipeline_gpt_instruct",
        #                                           hierarchical_config=HierarchicalConfig(),
        #                                           metadata_extraction=MetadataExtractionConfig(model="gpt-3.5-turbo-instruct"))
        # #
        # node_creator = TextNodeCreator(node_config=node_creation_config)
        # #
        # text_nodes = node_creator.create_or_load_nodes()
        # #
        embed_model = get_embed_model()

        index_manager = VectorIndexManager(embed_model=embed_model)
            ## To use new text nodes:
        # index = index_manager.get_or_create_index(text_nodes)
            ## Otherwise we'll use saved nodes:
        index = index_manager.get_or_create_index()

        mock_attributes = [
            {
                "attribute_id": "bg6t6xhlfxbw",
                "attribute_group_id": "qbxtcrbchk2v",
                "name": "Weight",
                "description": "The weight of the equipment's frame",
                "unit": "pounds",
                "format": "number",
                "vocabulary_options": [],
                "attribute_group_id": "askfdjsrlj",
                "attribute_group_name": "Information"
            },
            {
                "attribute_id": "bg6t6xhlfxby",
                "attribute_group_id": "qbxtcrbchk2v",
                "name": "Max dig depth",
                "description": "",
                "unit": "inches",
                "format": "number",
                "vocabulary_options": ["Diesel", "Electric"],
                "attribute_group_id": "askfdtsrlj",
                "attribute_group_name": "Information"
            },
            {
                "attribute_id": "bnccqpc9nwtd",
                "attribute_group_id": "qbxtcrbchk2v",
                "name": "Introduced in 1995",
                "description": "",
                "unit": "boolean",
                "format": "string",
                "vocabulary_options": [],
                "attribute_group_id": "askfdjsalj",
                "attribute_group_name": "Information"
            }
        ]

        # Initialize with just the vector store and LLM model as dependencies
        query_builder = StructuredQueryEngineBuilder(index= index, llm_model="gpt-4o-mini")

        # Set up your query parameters
        prompt_template = "Add attributes to model if found, or None for a {make} {model}"


            # For a single query
        single_query_pairs = [("Baretto", "1324D Standard Trencher")]

            # For multiple queries
        make_model_pairs = [
            ("Baretto", "1324D Standard Trencher"),
            ("Baretto", "2024RTK Track Trencher"),
            ("Baretto", "1324STK Track Trencher")
        ]

        # Query with arguments listed as explicit dependencies
        response = await query_builder.aquery(
            make_model_pairs=make_model_pairs,  # or single_query_pairs for one item
            attributes=mock_attributes,
            prompt_template=prompt_template
        )

        print(json.dumps(response, indent=2))

        # Convert to CSV
        csv_data = CSVCreator.export_dict_to_csv(data=response)
        print(csv_data.decode('utf-8'))

    except Exception as e:
        return error_handler.handle_error(e)

    return 0


# Main execution block - This is where we orchestrate everything
if __name__ == "__main__":
    ## This is Python and Asyncio's version of swift's Task { } essentially.
    ## Make an async call from a syncronous context
    asyncio.run(main())