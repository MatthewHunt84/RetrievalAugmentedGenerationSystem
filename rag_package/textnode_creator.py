from pathlib import Path
import json
import pickle
import time
import re
from datetime import datetime
import logging
from typing import List, Dict, Optional, Any, Tuple, Sequence
from dataclasses import dataclass
from llama_index.core.schema import TextNode, Document, BaseNode
from pydantic import Field, BaseModel
from llama_index.core.node_parser import NodeParser
from rag_package.errors import TextNodeCreationError
from rag_package.rag_config import NodeCreationConfig


class MarkdownHeaderSplitter(NodeParser, BaseModel):
    """
    A specialized node parser that splits text based on markdown header levels.
    By inheriting from both NodeParser and BaseModel, we ensure compatibility
    with LlamaIndex's internal structure while maintaining our custom functionality.
    """
    # Define the fields using Pydantic's Field
    header_level: int = Field(
        description="The header level to split on (1 for h1, 2 for h2, etc.)"
    )
    chunk_size: int = Field(
        description="Maximum size of text chunks"
    )
    chunk_overlap: int = Field(
        description="Number of characters to overlap between chunks"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in nodes"
    )
    include_prev_next_rel: bool = Field(
        default=True,
        description="Whether to include previous/next relationships"
    )

    class Config:
        # This tells Pydantic to be flexible with extra attributes
        arbitrary_types_allowed = True

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any
    ) -> List[BaseNode]:
        """
        Required implementation of NodeParser's abstract method.
        Acts as a pass-through since we handle the main processing in get_nodes_from_documents.
        """
        return list(nodes)

    def _split_text_with_headers(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Split text into sections based on markdown headers.

        This method analyzes the text content and breaks it apart at header boundaries,
        maintaining the hierarchical structure of the document. Each section includes
        its header and all content until the next header of the same or higher level.

        Returns:
            List of tuples containing:
            - Header level (number of # characters)
            - Header text (without the # marks)
            - Section content (including the header)
        """
        lines = text.split('\n')
        sections = []
        current_section = []
        current_header_level = 0
        current_header = ""

        for line in lines:
            # Look for markdown headers (e.g., "# Header" or "### Subheader")
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())

            if header_match:
                # If we have accumulated content in current_section, save it
                if current_section:
                    sections.append((
                        current_header_level,
                        current_header,
                        '\n'.join(current_section)
                    ))
                    current_section = []

                # Begin a new section with this header
                current_header_level = len(header_match.group(1))
                current_header = header_match.group(2)
                current_section = [line]
            else:
                # Add non-header lines to the current section
                current_section.append(line)

        # Don't forget to add the last section
        if current_section:
            sections.append((
                current_header_level,
                current_header,
                '\n'.join(current_section)
            ))

        return sections

    def _create_node_from_section(
            self,
            text: str,
            metadata: dict,
            header_info: Tuple[int, str]
    ) -> BaseNode:
        """
        Create a node from a section of text with header information.

        This method creates a TextNode instance that includes both the content
        and metadata about its place in the document hierarchy. The metadata
        helps maintain relationships between different sections of the document.
        """
        header_level, header_text = header_info

        # Combine existing metadata with header information
        enhanced_metadata = {
            **metadata,
            "header_info": {
                "level": header_level,
                "text": header_text
            }
        }

        return TextNode(
            text=text,
            metadata=enhanced_metadata,
            relationships={}  # Will be populated later with parent-child relationships
        )

    def get_nodes_from_documents(
            self,
            documents: List[Document],
            show_progress: bool = False
    ) -> List[BaseNode]:
        """
        Parse documents into nodes based on markdown structure.

        This is the main entry point for document processing. It analyzes each
        document's structure and creates a hierarchical representation using nodes.
        Each node corresponds to a section of the document at the appropriate
        header level.
        """
        nodes = []

        for doc in documents:
            # Split each document into sections based on headers
            sections = self._split_text_with_headers(doc.text)

            # Process sections that match our target header level
            for header_level, header_text, section_text in sections:
                # Check if this section should be processed by this parser level
                if (header_level == self.header_level or
                        (self.header_level == 1 and header_level == 0)):
                    node = self._create_node_from_section(
                        text=section_text,
                        metadata=doc.metadata or {},
                        header_info=(header_level, header_text)
                    )
                    nodes.append(node)

        return nodes


class TextNodeCreator:
    """
    Enhanced text node creator that processes markdown documents using a hierarchical approach.
    This implementation respects document structure and maintains hierarchical relationships
    between sections of text.
    """

    def __init__(self, node_config: NodeCreationConfig):
        """Initialize the creator with configuration settings."""
        self.config = node_config
        self.parsed_results_path = Path(node_config.parsed_results_path)
        self.output_dir = Path(node_config.output_dir)
        self.analysis_dir = Path("analysis")

        # Initialize directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._setup_logging()

        # Initialize hierarchical parser with markdown awareness
        self.parsers = self._initialize_parsers()

    def _initialize_parsers(self) -> Dict[str, MarkdownHeaderSplitter]:
        """
        Initialize a set of markdown-aware parsers for different header levels.
        Each parser is configured to handle a specific level of document hierarchy.
        """
        chunk_sizes = self.config.chunk_sizes
        chunk_overlap = self.config.chunk_overlap

        parsers = {}
        # Create parsers for different header levels (h1, h2, h3, etc.)
        for level in range(1, len(chunk_sizes) + 1):
            parser = MarkdownHeaderSplitter(
                header_level=level,
                chunk_size=chunk_sizes[level - 1],  # -1 because levels start at 1
                chunk_overlap=chunk_overlap,
                include_metadata=True,
                include_prev_next_rel=True
            )
            parsers[f"h{level}_sections"] = parser

        return parsers

    def _process_document_content(
            self,
            content: str,
            base_metadata: dict,
    ) -> List[BaseNode]:
        """
        Process document content using markdown-aware parsing.

        This method processes the document through multiple levels of parsing,
        establishing relationships between sections at different levels.
        """
        doc = Document(text=content, metadata=base_metadata)
        all_nodes = []

        # Process document with each parser level
        for parser_name, parser in self.parsers.items():
            level_nodes = parser.get_nodes_from_documents([doc])

            # Enhance nodes with level information
            for node in level_nodes:
                header_level = node.metadata["header_info"]["level"]
                node.metadata["hierarchy_info"] = {
                    "level": header_level - 1,  # Convert to 0-based level
                    "parser": parser_name
                }

            all_nodes.extend(level_nodes)

        # Establish parent-child relationships
        self._establish_relationships(all_nodes)

        return all_nodes

    def _establish_relationships(self, nodes: List[BaseNode]) -> None:
        """
        Establish hierarchical relationships between nodes based on their header levels.

        This method creates parent-child relationships between sections based on their
        header levels and position in the document. It properly handles the set-based
        relationship structure that LlamaIndex expects.
        """
        # Sort nodes by their position in the document
        sorted_nodes = sorted(
            nodes,
            key=lambda n: (
                n.metadata["document_info"]["page_num"],
                n.text.find(n.metadata["header_info"]["text"])
            )
        )

        for i, node in enumerate(sorted_nodes):
            current_level = node.metadata["header_info"]["level"]

            # Look backwards for the nearest higher-level node
            for previous_node in reversed(sorted_nodes[:i]):
                previous_level = previous_node.metadata["header_info"]["level"]

                if previous_level < current_level:
                    # Initialize relationship sets if they don't exist
                    if "parent" not in node.relationships:
                        node.relationships["parent"] = set()
                    if "child" not in previous_node.relationships:
                        previous_node.relationships["child"] = set()

                    # Add relationships using set operations
                    node.relationships["parent"].add(previous_node.node_id)
                    previous_node.relationships["child"].add(node.node_id)
                    break


    def _get_parent_id(self, node: BaseNode) -> Optional[str]:
        """
        Get the parent ID from a node's relationships.

        This method safely handles the set-based relationship structure that LlamaIndex uses.
        Rather than trying to access the parent set with an index, we retrieve the first
        element if it exists.

        Args:
            node: The node whose parent we want to find

        Returns:
            The parent node ID if one exists, None otherwise
        """
        parent_ids = node.relationships.get("parent", set())
        # Convert to list and get first element if exists, otherwise return None
        return list(parent_ids)[0] if parent_ids else None

    def create_nodes(self) -> List[BaseNode]:
        """
        Create hierarchical nodes from parsed markdown content.
        """
        start_time = time.time()

        try:
            self.logger.info("Beginning hierarchical node creation process...")
            with open(self.parsed_results_path, 'r', encoding='utf-8') as f:
                md_json_objs = json.load(f)

            all_nodes = []
            for result in md_json_objs:
                document_name = Path(result["file_path"]).name
                self.logger.info(f"Processing document: {document_name}")

                for idx, page_dict in enumerate(result["pages"]):
                    base_metadata = {
                        "pipeline_info": self.config.base_metadata["pipeline_info"],
                        "document_info": {
                            "name": document_name,
                            "total_pages": len(result["pages"]),
                            "page_num": idx + 1
                        }
                    }

                    # Process content using hierarchical parser
                    nodes = self._process_document_content(
                        page_dict["md"],
                        base_metadata
                    )
                    all_nodes.extend(nodes)
                    self.logger.info(
                        f"Created {len(nodes)} nodes for page {idx + 1}"
                    )

            if all_nodes:
                # self._save_outputs(all_nodes)
                self.analyze_node_hierarchy(all_nodes)

            execution_time = time.time() - start_time
            self._log_execution_time(execution_time)

            return all_nodes

        except Exception as e:
            self.logger.error(f"Hierarchical node creation failed: {str(e)}")
            raise TextNodeCreationError(f"Failed to create nodes: {str(e)}")

    def analyze_node_hierarchy(self, nodes: List[BaseNode]) -> None:
        """
        Analyze the hierarchical structure of nodes, providing detailed information about
        nodes from levels 0, 1, and 2 on page 10. This helps understand how content is
        organized and how different levels relate to each other on the page.
        """
        analysis_path = self.analysis_dir / f"{self.config.pipeline_name}_hierarchy_analysis.txt"

        with open(analysis_path, 'w', encoding='utf-8') as f:
            # First, create our overall summary of nodes across all pages
            nodes_by_level = {}
            for node in nodes:
                level = node.metadata["hierarchy_info"]["level"]
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)

            # Write the summary statistics section
            f.write("=== Node Hierarchy Analysis ===\n\n")
            f.write(f"Total Nodes: {len(nodes)}\n")
            f.write("Nodes per level:\n")
            for level in sorted(nodes_by_level.keys()):
                count = len(nodes_by_level[level])
                f.write(f"Level {level}: {count} nodes\n")
            f.write("\n")

            # Now analyze each level for page 10
            for level in [0, 1, 2]:
                f.write(f"\n=== Level {level} Nodes from Page 10 ===\n\n")

                # Filter nodes for the current level and page 10
                page_10_level_nodes = [
                    node for node in nodes
                    if (node.metadata["hierarchy_info"]["level"] == level and
                        node.metadata["document_info"]["page_num"] == 10)
                ]

                if not page_10_level_nodes:
                    f.write(f"No level {level} nodes found on page 10.\n")
                else:
                    # Convert node information to JSON-friendly format
                    node_data = []
                    for node in page_10_level_nodes:
                        # Create a clean dictionary representation of the node
                        node_info = {
                            "node_id": node.node_id,
                            "text": node.text,
                            "metadata": {
                                "header_info": node.metadata.get("header_info", {}),
                                "document_info": node.metadata.get("document_info", {}),
                                "hierarchy_info": node.metadata.get("hierarchy_info", {})
                            },
                            "relationships": {
                                "parents": list(node.relationships.get("parent", set())),
                                "children": list(node.relationships.get("child", set()))
                            }
                        }
                        node_data.append(node_info)

                    # Write the JSON representation with proper formatting
                    import json
                    f.write(json.dumps(node_data, indent=2, ensure_ascii=False))

                    # Add a summary for this level
                    f.write(f"\n\nFound {len(page_10_level_nodes)} level {level} nodes on page 10\n")

                    # Add a visual separator between levels
                    f.write("\n" + "=" * 80 + "\n")

            # Log a summary of what we found
            for level in [0, 1, 2]:
                level_count = len([
                    node for node in nodes
                    if (node.metadata["hierarchy_info"]["level"] == level and
                        node.metadata["document_info"]["page_num"] == 10)
                ])
                self.logger.info(f"Found {level_count} level {level} nodes on page 10")

            self.logger.info(f"Detailed analysis has been written to {analysis_path}")

    def _get_hierarchy_statistics(self, nodes: List[BaseNode]) -> Dict[str, Any]:
        """
        Calculate statistics about the node hierarchy.

        This method analyzes the distribution of nodes across different levels
        and calculates metrics about parent-child relationships. It's important
        to handle relationships consistently as sets throughout the analysis.
        """
        level_counts = {}
        parent_child_counts = {}

        for node in nodes:
            # Count nodes per level
            level = node.metadata["hierarchy_info"]["level"]
            level_counts[level] = level_counts.get(level, 0) + 1

            # Get child relationships, using set as default for consistency
            children = node.relationships.get("child", set())
            parent_child_counts[node.node_id] = len(children)

        return {
            "nodes_per_level": level_counts,
            "average_children_per_node": sum(parent_child_counts.values()) / len(nodes),
            "max_children_per_node": max(parent_child_counts.values(), default=0)
        }

    def _setup_logging(self):
        """Configure logging to write to both a file and the console."""
        # Clear any existing handlers
        self.logger.handlers = []

        # Create file handler
        log_file = self.analysis_dir / "node_creation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _log_execution_time(self, execution_time: float) -> None:
        """Log execution time with hierarchical processing details."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timing_log = (
            f"{timestamp} - Pipeline: {self.config.pipeline_name}, "
            f"Hierarchical Processing Time: {execution_time:.2f} seconds\n"
        )

        with open(self.analysis_dir / "node_creation_times.txt", 'a') as f:
            f.write(timing_log)