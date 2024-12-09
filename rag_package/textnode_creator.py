from pathlib import Path
import json
import re
from typing import List, Dict, Any
import pickle
import logging

from llama_index.core.schema import TextNode, Document
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from rag_package.errors import TextNodeCreationError


class TextNodeCreator:
    """
    Enhanced version of TextNodeCreator that uses configuration injection
    for processing product catalogs with hierarchical structure and rich
    metadata extraction.
    """

    def __init__(self, node_config):
        """
        Initialize the TextNodeCreator with configuration settings.

        Args:
            node_config: NodeCreationConfig instance containing all necessary settings
        """
        self.config = node_config
        self.parsed_results_path = Path(node_config.parsed_results_path)
        self.image_dir = Path(node_config.image_dir)
        self.output_dir = Path(node_config.output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            print("Creating node parsers and setting up hierarchy...")

            # Create sentence splitters for each level
            level1_splitter = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            level2_splitter = SentenceSplitter(
                chunk_size=self.config.chunk_size // 2,
                chunk_overlap=self.config.chunk_overlap
            )

            # Define parser IDs and map
            node_parser_ids = ["level_1", "level_2"]
            node_parser_map = {
                "level_1": level1_splitter,
                "level_2": level2_splitter
            }

            print("Initializing HierarchicalNodeParser...")
            self.hierarchical_parser = HierarchicalNodeParser(
                node_parser_ids=node_parser_ids,
                node_parser_map=node_parser_map
            )
            print("Successfully initialized HierarchicalNodeParser")

            # Initialize extractors
            print("Initializing metadata extractors...")
            self.title_extractor = TitleExtractor(nodes=5)
            self.qa_extractor = QuestionsAnsweredExtractor()
            self.summary_extractor = SummaryExtractor(summaries=["self", "prev", "next"])
            print("Successfully initialized extractors")

        except Exception as e:
            print(f"Initialization error details: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise TextNodeCreationError(f"Failed to initialize parser: {str(e)}")

        # Validate paths
        if not self.parsed_results_path.exists():
            raise TextNodeCreationError(
                f"Parsed results file not found at {self.parsed_results_path}"
            )
        if not self.image_dir.exists():
            raise TextNodeCreationError(
                f"Image directory not found at {self.image_dir}"
            )

    def _apply_extractors(self, nodes: List[TextNode]) -> List[TextNode]:
        """Apply metadata extractors to nodes."""
        try:
            # Apply each extractor sequentially
            for node in nodes:
                try:
                    # Extract titles
                    title_response = self.title_extractor.process_nodes([node])
                    node.metadata.update(title_response[0].metadata)

                    # Extract potential questions
                    qa_response = self.qa_extractor.process_nodes([node])
                    node.metadata.update(qa_response[0].metadata)

                    # Extract summaries (self, prev, and next contexts)
                    summary_response = self.summary_extractor.process_nodes([node])
                    node.metadata.update(summary_response[0].metadata)

                    # Add machinery-related keyword flags directly to metadata
                    text_lower = node.text.lower()
                    node.metadata["machinery_keywords"] = {
                        "has_capacity_info": any(
                            word in text_lower for word in ["capacity", "can hold", "maximum", "load"]),
                        "has_weight_info": any(
                            word in text_lower for word in ["weight", "kg", "tons", "pounds", "lbs"]),
                        "has_dimension_info": any(word in text_lower for word in
                                                  ["dimensions", "length", "width", "height", "mm", "cm", "meters"]),
                        "has_power_info": any(
                            word in text_lower for word in ["power", "hp", "kw", "horsepower", "watts"]),
                        "has_engine_info": any(
                            word in text_lower for word in ["engine", "motor", "diesel", "gasoline", "fuel", "rpm"]),
                        "has_hydraulic_info": any(
                            word in text_lower for word in ["hydraulic", "pressure", "flow", "psi", "bar"]),
                        "has_electrical_info": any(
                            word in text_lower for word in ["voltage", "electrical", "battery", "volts", "amps"]),
                        "has_safety_info": any(
                            word in text_lower for word in ["safety", "warning", "caution", "danger", "protective"])
                    }

                except Exception as e:
                    print(f"Warning: Error processing extractors for node: {str(e)}")
                    continue

            return nodes
        except Exception as e:
            print(f"Warning: Error during metadata extraction: {str(e)}")
            return nodes

    def _process_content(self, content: str, metadata: Dict[str, Any]) -> List[TextNode]:
        """Process a single piece of content into nodes."""
        try:
            # Create a Document object first
            doc = Document(text=content, metadata=metadata)

            # Process through hierarchical parser
            print(f"Processing document of length {len(content)} characters...")
            nodes = self.hierarchical_parser.get_nodes_from_documents(
                documents=[doc],
                show_progress=True
            )
            print(f"Created {len(nodes)} initial nodes")

            # Apply metadata extractors
            nodes = self._apply_extractors(nodes)

            # Add hierarchical relationship metadata
            for node in nodes:
                if isinstance(node, TextNode):
                    node.metadata.update({
                        "hierarchy_level": node.metadata.get("hierarchy_level", 0),
                        "parent_title": node.metadata.get("parent_title", ""),
                        "section_title": node.metadata.get("title", ""),
                    })

            return nodes
        except Exception as e:
            print(f"Error processing content: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []

    # ... rest of the methods remain the same ...

    def create_nodes(self) -> List[TextNode]:
        """
        Create hierarchical nodes from parsed documents with enhanced metadata extraction.

        Returns:
            List of TextNode objects with hierarchical structure and rich metadata
        """
        try:
            print("Loading parsed document results...")
            with open(self.parsed_results_path, 'r', encoding='utf-8') as f:
                md_json_objs = json.load(f)

            all_nodes = []
            image_files = sorted(
                [f for f in self.image_dir.iterdir() if f.is_file()],
                key=lambda x: int(
                    re.search(r"-page-(\d+)\.jpg$", str(x)).group(1) if re.search(r"-page-(\d+)\.jpg$", str(x)) else 0)
            )

            for result in md_json_objs:
                document_name = Path(result["file_path"]).name
                print(f"Processing document: {document_name}")

                for idx, page_dict in enumerate(result["pages"]):
                    print(f"Processing page {idx + 1}")

                    metadata = {
                        "pipeline_name": self.config.pipeline_name,
                        "image_path": str(image_files[idx]),
                        "page_num": idx + 1,
                        "document_name": document_name,
                        "total_pages": len(result["pages"]),
                        "chunk_size": self.config.chunk_size,
                        "chunk_overlap": self.config.chunk_overlap
                    }

                    nodes = self._process_content(
                        content=page_dict["md"],
                        metadata=metadata
                    )

                    all_nodes.extend(nodes)
                    print(f"Created {len(nodes)} nodes for page {idx + 1}")

            print(f"\nCreated {len(all_nodes)} total nodes")

            # Save the nodes
            if all_nodes:
                output_path = self.output_dir / f"{self.config.pipeline_name}_nodes.pkl"
                with open(output_path, 'wb') as f:
                    pickle.dump(all_nodes, f)

                # Save a summary in JSON format for easy inspection
                summary_path = self.output_dir / f"{self.config.pipeline_name}_summary.json"
                summary = {
                    "total_nodes": len(all_nodes),
                    "pipeline_config": {
                        "chunk_size": self.config.chunk_size,
                        "chunk_overlap": self.config.chunk_overlap,
                        "pipeline_name": self.config.pipeline_name
                    },
                    "node_preview": [
                        {
                            "text_preview": str(node.text)[:100] + "...",
                            "metadata": node.metadata,
                            "keywords": node.metadata.get("keywords", []),
                            "summary": node.metadata.get("summary", ""),
                            "title": node.metadata.get("title", ""),
                            "questions_answered": node.metadata.get("questions_answered", [])
                        }
                        for node in all_nodes[:5]  # Preview first 5 nodes
                    ]
                }

                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)

                print(f"Saved nodes to {output_path}")
                print(f"Saved summary to {summary_path}")

            return all_nodes

        except Exception as e:
            raise TextNodeCreationError(f"Failed to create nodes: {str(e)}")