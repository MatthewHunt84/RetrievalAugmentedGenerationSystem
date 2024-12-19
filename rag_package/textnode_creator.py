from pathlib import Path
import json
import time
import re
import logging
from datetime import datetime
from typing import Sequence
from llama_index.core.schema import TextNode, Document, BaseNode
from pydantic import Field, BaseModel
from llama_index.core.node_parser import NodeParser
from rag_package.errors import TextNodeCreationError
from rag_package.rag_config import NodeCreationConfig, EquipmentSpecs

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
        **kwargs: any
    ) -> list[BaseNode]:
        """
        Required implementation of NodeParser's abstract method.
        Acts as a pass-through since we handle the main processing in get_nodes_from_documents.
        """
        return list(nodes)

    def _split_text_with_headers(self, text: str) -> list[tuple[int, str, str]]:
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
            header_info: tuple[int, str]
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
            documents: list[Document],
            show_progress: bool = False
    ) -> list[BaseNode]:
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

    def _initialize_parsers(self) -> dict[str, MarkdownHeaderSplitter]:
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
    ) -> list[BaseNode]:
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

    def _establish_relationships(self, nodes: list[BaseNode]) -> None:
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

    def _get_parent_id(self, node: BaseNode) -> str | None:
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

    def create_nodes(self) -> list[BaseNode]:
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
            """
            NEW (Metadata Extraction): Metadata extraction
            """
            if all_nodes:
                self._enhance_page_10_nodes_with_metadata(all_nodes)
            #     # Extract document-level metadata
            #     doc_metadata = self._extract_document_metadata(
            #         "\n".join(node.text for node in all_nodes)
            #     )
            #
            #     # Process nodes in batches
            #     model_metadata = {}
            #     for batch in self._batch_model_descriptions(all_nodes):
            #         batch_metadata = self._extract_batch_metadata(batch)
            #         model_metadata.update(batch_metadata)
            #
            #     # Enhance nodes with metadata
            #     self._enhance_nodes_with_metadata(
            #         all_nodes, doc_metadata, model_metadata
            #     )

            if all_nodes:
                self.analyze_node_hierarchy(all_nodes)

            execution_time = time.time() - start_time
            self._log_execution_time(execution_time)

            return all_nodes

        except Exception as e:
            self.logger.error(f"Hierarchical node creation failed: {str(e)}")
            raise TextNodeCreationError(f"Failed to create nodes: {str(e)}")

    def analyze_node_hierarchy(self, nodes: list[BaseNode]) -> None:
        """
        Analyze the hierarchical structure of nodes, providing detailed information about
        nodes from levels 0, 1, and 2 on page 10. This helps understand how content is
        organized and how different levels relate to each other on the page.
        """
        analysis_path = self.analysis_dir / f"{self.config.pipeline_name}_hierarchy_analysis.txt"

        with open(analysis_path, 'w', encoding='utf-8') as f:
            # ... [existing summary code remains the same] ...

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
                                "hierarchy_info": node.metadata.get("hierarchy_info", {}),
                                # Add new metadata fields
                                "document_metadata": node.metadata.get("document_metadata", {}),
                                "equipment_specs": node.metadata.get("equipment_specs", {}),
                                "extraction_info": node.metadata.get("extraction_info", {})
                            },
                            "relationships": {
                                "parents": list(node.relationships.get("parent", set())),
                                "children": list(node.relationships.get("child", set()))
                            }
                        }
                        node_data.append(node_info)

                    # Write the JSON representation with proper formatting
                    f.write(json.dumps(node_data, indent=2, ensure_ascii=False))

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

    """
    NEW (Metadata Extraction): New methods
    """

    def _batch_model_descriptions(self, nodes: list[BaseNode]) -> list[list[BaseNode]]:
        """
        Group similar model descriptions together for batch processing.

        Args:
            nodes: List of nodes to process

        Returns:
            List of node batches, where each batch contains related models
        """
        batches = []
        current_batch = []
        max_batch_size = self.config.metadata_extraction.batch_size

        for node in nodes:
            # Start new batch if current one is full
            if len(current_batch) >= max_batch_size:
                batches.append(current_batch)
                current_batch = []

            current_batch.append(node)

        # Add any remaining nodes
        if current_batch:
            batches.append(current_batch)

        return batches

    def _extract_document_metadata(self, doc_content: str) -> dict[str, any]:
        """
        Extract document-level metadata using LLM.
        """
        prompt = f"""
        Analyze the following text and extract metadata.

        YOU MUST RESPOND WITH ONLY A VALID JSON OBJECT using this exact structure:
        {{
            "primary_categories": [list of equipment categories found],
            "manufacturer": "main manufacturer name",
            "document_type": "type of document (e.g., catalog, brochure)",
            "document_date": "date if mentioned"
        }}

        Text to analyze:
        {doc_content}
        """

        try:
            client = self.config.metadata_extraction.get_client()
            message = client.messages.create(
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                model=self.config.metadata_extraction.model_name,
                max_tokens=1000,
                temperature=self.config.metadata_extraction.temperature,
            )

            if message.content:
                try:
                    # Extract text from TextBlock list
                    if isinstance(message.content, list) and message.content:
                        text_content = message.content[0].text
                    else:
                        text_content = str(message.content)

                    self.logger.debug(f"Attempting to parse JSON: {text_content}")
                    parsed_data = json.loads(text_content)
                    self.logger.debug(f"Successfully parsed JSON: {parsed_data}")
                    return parsed_data
                except Exception as e:
                    self.logger.error(f"Failed to process LLM response: {str(e)}")
                    self.logger.error(f"Raw response was: {message.content}")
                    return {}
            return {}
        except Exception as e:
            self.logger.error(f"Document metadata extraction failed: {str(e)}")
            self.logger.error(f"Full error: {str(e.__class__.__name__)}")
            return {}

    def _extract_batch_metadata(
            self,
            batch: list[BaseNode]
    ) -> dict[str, EquipmentSpecs]:
        """
        Extract metadata for a batch of model descriptions.
        """
        batch_text = "\n\n".join(node.text for node in batch)
        prompt = f"""
        Extract metadata for each equipment model in the following text.

        For each model, create a JSON object with this exact structure:
        {{
            "product_name": "full product name",
            "model_number": "specific model identifier",
            "product_category": "equipment category",
            "manufacturer": "manufacturer name",
            "specifications": {{
                "dimensions": {{
                    "size relevant measurements"
                }},
                "engine": {{
                    "engine specifications"
                }},
                "performance": {{
                    "performance metrics"
                }}
            }}
        }}

        Return an array of these objects, one for each model found.

        Text to analyze:
        {batch_text}
        """

        try:
            client = self.config.metadata_extraction.get_client()
            message = client.messages.create(
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                model=self.config.metadata_extraction.model_name,
                max_tokens=2000,
                temperature=self.config.metadata_extraction.temperature,
            )

            if message.content:
                try:
                    # Extract text from TextBlock list
                    if isinstance(message.content, list) and message.content:
                        text_content = message.content[0].text
                    else:
                        text_content = str(message.content)

                    self.logger.debug(f"Raw LLM response text: {text_content}")
                    self.logger.debug(f"Attempting to parse JSON: {text_content}")
                    metadata_list = json.loads(text_content)
                    self.logger.debug(f"Successfully parsed JSON: {metadata_list}")
                    # Clean up any explanatory text before the JSON
                    if '[' in text_content:
                        text_content = text_content[text_content.find('['):text_content.rfind(']') + 1]

                    metadata_list = json.loads(text_content)
                    if isinstance(metadata_list, list):
                        return self._match_metadata_to_nodes(batch, metadata_list)
                    return {}
                except Exception as e:
                    self.logger.error(f"Failed to process LLM response: {str(e)}")
                    self.logger.error(f"Raw response was: {message.content}")
                    return {}
            return {}
        except Exception as e:
            self.logger.error(f"Batch metadata extraction failed: {str(e)}")
            self.logger.error(f"Full error: {str(e.__class__.__name__)}")
            return {}

    def _match_metadata_to_nodes(
            self,
            nodes: list[BaseNode],
            metadata_list: list[dict[str, any]]
    ) -> dict[str, EquipmentSpecs]:
        """
        Match extracted metadata to the correct nodes.

        Args:
            nodes: List of nodes
            metadata_list: List of extracted metadata dictionaries

        Returns:
            Dictionary mapping node IDs to matched metadata
        """
        matched_metadata = {}

        for node in nodes:
            node_text = node.text.lower()

            # Find best matching metadata based on model number
            best_match = None
            best_score = 0

            for metadata in metadata_list:
                if not metadata.get("model_number"):
                    continue

                model_num = metadata["model_number"].lower()
                if model_num in node_text:
                    # Simple matching score based on text overlap
                    score = len(model_num) / len(node_text)
                    if score > best_score:
                        best_score = score
                        best_match = metadata

            if best_match and best_score > self.config.metadata_extraction.confidence_threshold:
                matched_metadata[node.node_id] = EquipmentSpecs(**best_match)

        return matched_metadata

    def _enhance_nodes_with_metadata(
            self,
            nodes: list[BaseNode],
            doc_metadata: dict[str, any],
            model_metadata: dict[str, EquipmentSpecs]
    ) -> None:
        """
        Enhance nodes with extracted metadata.

        Args:
            nodes: List of nodes to enhance
            doc_metadata: Document-level metadata
            model_metadata: Model-specific metadata
        """
        timestamp = datetime.now().isoformat()

        for node in nodes:
            # Add document-level metadata to all nodes
            node.metadata["document_metadata"] = doc_metadata
            node.metadata["extraction_info"]["extraction_timestamp"] = timestamp

            # Add model-specific metadata if available
            if node.node_id in model_metadata:
                node.metadata["equipment_specs"] = model_metadata[node.node_id].model_dump()



    """
    Test Method, may be deleted also
    """

    def _enhance_page_10_nodes_with_metadata(
            self,
            nodes: list[BaseNode]
    ) -> None:
        """
        Test method to enhance only page 10 nodes with metadata.
        """
        # Filter for page 10 nodes
        page_10_nodes = [
            node for node in nodes
            if node.metadata["document_info"]["page_num"] == 10
        ]

        if not page_10_nodes:
            self.logger.info("No page 10 nodes found")
            return

        self.logger.info(f"Found {len(page_10_nodes)} nodes on page 10")

        try:
            # Extract document-level metadata for page 10 only
            raw_doc_metadata = self._extract_document_metadata(
                "\n".join(node.text for node in page_10_nodes)
            )

            # Add debug logging to see what we're trying to parse
            self.logger.debug(f"Raw doc metadata type: {type(raw_doc_metadata)}")
            self.logger.debug(f"Raw doc metadata content: {raw_doc_metadata}")

            # No need to parse if it's already a dict
            if isinstance(raw_doc_metadata, dict):
                doc_metadata = raw_doc_metadata
            else:
                doc_metadata = json.loads(str(raw_doc_metadata))

            self.logger.info("Successfully extracted document metadata for page 10")
            self.logger.debug(f"Document metadata: {json.dumps(doc_metadata, indent=2)}")

            # Process page 10 nodes in batches
            model_metadata = {}
            for batch in self._batch_model_descriptions(page_10_nodes):
                raw_batch_metadata = self._extract_batch_metadata(batch)

                # Add debug logging for batch metadata
                self.logger.debug(f"Raw batch metadata type: {type(raw_batch_metadata)}")
                self.logger.debug(f"Raw batch metadata content: {raw_batch_metadata}")

                # If it's already a dict, use it directly
                if isinstance(raw_batch_metadata, dict):
                    batch_metadata = raw_batch_metadata
                else:
                    try:
                        batch_metadata = json.loads(str(raw_batch_metadata))
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse batch metadata: {str(e)}")
                        batch_metadata = {}

                model_metadata.update(batch_metadata)
                self.logger.info(f"Processed batch of {len(batch)} nodes")
                self.logger.debug(f"Batch metadata: {json.dumps(batch_metadata, indent=2)}")

            # Enhance only page 10 nodes
            timestamp = datetime.now().isoformat()
            for node in page_10_nodes:
                # Initialize extraction_info if it doesn't exist
                if 'extraction_info' not in node.metadata:
                    node.metadata['extraction_info'] = {}

                # Add document-level metadata to all nodes
                node.metadata["document_metadata"] = doc_metadata
                node.metadata["extraction_info"]["extraction_timestamp"] = timestamp

                # Add model-specific metadata if available
                if node.node_id in model_metadata:
                    if isinstance(model_metadata[node.node_id], dict):
                        node.metadata["equipment_specs"] = model_metadata[node.node_id]
                    else:
                        node.metadata["equipment_specs"] = model_metadata[node.node_id].model_dump()

            self.logger.info("Successfully enhanced page 10 nodes with metadata")

        except Exception as e:
            self.logger.error(f"Page 10 metadata enhancement failed: {str(e)}")
            self.logger.error(f"Full error: {str(e.__class__.__name__)}")
            self.logger.error(f"Error occurred while processing metadata. Raw metadata content: {raw_doc_metadata}")
