import uuid
from pathlib import Path
import json
import time
import logging
from datetime import datetime
from llama_index.core.schema import Document, BaseNode
from rag_package.errors import TextNodeCreationError
from rag_package.config.node_creation_config import NodeCreationConfig
from rag_package.parsers.markdown_header_splitter import MarkdownHeaderSplitter

class TextNodeCreator:
    """
    Enhanced text node creator that processes markdown documents using a hierarchical approach.
    This implementation respects document structure and maintains hierarchical relationships
    between sections of text.
    """

    def __init__(self, node_config: NodeCreationConfig):
        """Initialize the creator with configuration settings."""
        self.config = node_config

        # First, set up logging before we do anything else
        self.logger = logging.getLogger(__name__)
        self._setup_logging()  # This configures the logger with handlers and levels

        # Now we can use logging in subsequent initialization steps
        self.logger.info("Initializing TextNodeCreator")

        # Validate essential configuration
        if not isinstance(self.config.paths_config.parsed_results_path, Path):
            self.config.paths_config.parsed_results_path = Path(
                self.config.paths_config.parsed_results_path
            )
            self.logger.debug("Converted parsed_results_path to Path object")

        # Initialize paths
        self.parsed_results_path = self.config.paths_config.parsed_results_path
        self.output_dir = self.config.paths_config.output_dir
        self.analysis_dir = self.config.paths_config.analysis_dir
        self.logger.debug("Initialized path configurations")

        # Now we can safely ensure directories exist
        self._ensure_output_paths()

        # Initialize parsers
        self.parsers = self._initialize_parsers()
        self.logger.info("TextNodeCreator initialization complete")

    def _ensure_output_paths(self) -> None:
        """
        Ensure all required output directories exist.

        This method creates any missing directories needed for the pipeline's operation.
        It's called during initialization to prevent file operation errors later.
        """
        for path_name, path in [
            ("output_dir", self.output_dir),
            ("analysis_dir", self.analysis_dir)
        ]:
            try:
                path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {path}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {path_name} at {path}: {str(e)}")
                raise TextNodeCreationError(f"Failed to create required directory: {str(e)}")

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

    def create_nodes(
            self,
            metadata_extraction_test_pages: list[int] | None = None,
            node_analysis_pages: list[int] | None = None
    ) -> list[BaseNode]:
        """
        Create hierarchical nodes from parsed markdown content.

        Args:
            metadata_extraction_test_pages: List of page numbers to enhance with metadata,
                or None to enhance all pages
            node_analysis_pages: List of page numbers to include in analysis output,
                or None to analyze page 1 only
        """
        start_time = time.time()

        try:
            self.logger.info("Beginning hierarchical node creation process...")
            if not self.parsed_results_path.exists():
                raise TextNodeCreationError(
                    f"Parsed results file not found at {self.parsed_results_path}"
                )

            with open(self.parsed_results_path, 'r', encoding='utf-8') as f:
                try:
                    md_json_objs = json.load(f)
                except json.JSONDecodeError as e:
                    raise TextNodeCreationError(
                        f"Failed to parse results file: {str(e)}"
                    )

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
                            "page_num": idx + 1,
                            "document_uuid": str(uuid.uuid4()),
                            "ingestion_timestamp": datetime.now().isoformat()
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
                # Filter nodes for metadata enhancement
                metadata_nodes = self._filter_nodes_by_pages(
                    all_nodes,
                    metadata_extraction_test_pages
                )
                if metadata_nodes:
                    self._enhance_nodes_with_metadata(metadata_nodes)

                # Set default analysis pages to [1] if None
                if node_analysis_pages is None:
                    node_analysis_pages = [1]

                # Filter nodes for analysis
                analysis_nodes = self._filter_nodes_by_pages(
                    all_nodes,
                    node_analysis_pages
                )
                if analysis_nodes:
                    self.analyze_node_hierarchy(analysis_nodes)

            execution_time = time.time() - start_time
            self._log_execution_time(execution_time)

            return all_nodes

        except Exception as e:
            self.logger.error(f"Hierarchical node creation failed: {str(e)}")
            raise TextNodeCreationError(f"Failed to create nodes: {str(e)}")

    def analyze_node_hierarchy(self, nodes: list[BaseNode]) -> None:
        """
        Analyze the hierarchical structure of nodes, showing the actual TextNode objects
        rather than JSON representations.
        """

        analysis_path = self.analysis_dir / f"{self.config.pipeline_name}_raw_node_analysis.txt"

        if self.config.verbose:
            self.logger.info(f"Analyzing hierarchy for {len(nodes)} nodes")
            self.logger.info(f"Writing analysis to {analysis_path}")

        with open(analysis_path, 'w', encoding='utf-8') as f:
            # Write LLM calls summary
            f.write("=== LLM Calls Analysis ===\n")
            f.write(f"Document Metadata Extraction Calls: 1\n")
            num_batches = (
                                      len(nodes) + self.config.metadata_extraction.batch_size - 1) // self.config.metadata_extraction.batch_size
            f.write(f"Batch Metadata Extraction Calls: {num_batches}\n")
            f.write(f"Total LLM Calls: {1 + num_batches}\n\n")

            # Analyze each level
            for level in [0, 1, 2]:
                f.write(f"\n=== Level {level} Nodes ===\n\n")

                # Filter nodes for current level
                level_nodes = [
                    node for node in nodes
                    if node.metadata["hierarchy_info"]["level"] == level
                ]

                if not level_nodes:
                    f.write(f"No level {level} nodes found.\n")
                else:
                    for i, node in enumerate(level_nodes, 1):
                        f.write(f"Node {i}:\n")
                        f.write("=" * 50 + "\n")

                        # Display node attributes
                        f.write(f"Node ID: {node.node_id}\n")
                        f.write(f"Node Type: {node.__class__.__name__}\n")
                        f.write(f"Page Number: {node.metadata['document_info']['page_num']}\n")

                        # Display text content
                        f.write("\nText Content:\n")
                        f.write("-" * 20 + "\n")
                        f.write(f"{node.text}\n")

                        # Display relationships
                        f.write("\nRelationships:\n")
                        f.write("-" * 20 + "\n")
                        for rel_type, rel_ids in node.relationships.items():
                            f.write(f"{rel_type}: {sorted(list(rel_ids))}\n")

                        # Display metadata structure
                        f.write("\nMetadata Structure:\n")
                        f.write("-" * 20 + "\n")
                        for key in node.metadata:
                            if isinstance(node.metadata[key], dict):
                                f.write(f"{key}:\n")
                                for subkey in node.metadata[key]:
                                    f.write(f"  {subkey}: {type(node.metadata[key][subkey]).__name__}\n")
                            else:
                                f.write(f"{key}: {type(node.metadata[key]).__name__}\n")

                        # Display metadata values
                        f.write("\nMetadata Values:\n")
                        f.write("-" * 20 + "\n")
                        for key, value in node.metadata.items():
                            f.write(f"{key}:\n")
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    f.write(f"  {subkey}: {subvalue}\n")
                            else:
                                f.write(f"  {value}\n")

                        f.write("\n" + "=" * 50 + "\n\n")

                    # Summary statistics
                    f.write(f"\nTotal nodes at level {level}: {len(level_nodes)}\n")
                    f.write(
                        f"Nodes with equipment metadata: {sum(1 for n in level_nodes if any(n.metadata.get('equipment_metadata', {}).values()))}\n")
                    f.write(
                        f"Nodes with document metadata: {sum(1 for n in level_nodes if any(n.metadata.get('document_metadata', {}).values()))}\n")
                    f.write("\n" + "=" * 50 + "\n\n")

    def _setup_logging(self):
        """
        Configure logging based on the logging configuration.
        Supports both file and console logging with configurable formats.
        """
        # Clear any existing handlers
        self.logger.handlers = []

        if self.config.logging_config.log_to_file:
            # Create file handler using paths from config
            log_file = self.config.paths_config.analysis_dir / "node_creation.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.config.logging_config.level)
            file_handler.setFormatter(
                logging.Formatter(self.config.logging_config.format)
            )
            self.logger.addHandler(file_handler)

        if self.config.logging_config.log_to_console:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.config.logging_config.level)
            console_handler.setFormatter(
                logging.Formatter(self.config.logging_config.format)
            )
            self.logger.addHandler(console_handler)


    def _log_execution_time(self, execution_time: float) -> None:
        """Log execution time with hierarchical processing details."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Add mode only if it exists in config
        mode_info = f"Mode: {self.config.mode}, " if hasattr(self.config, 'mode') else ""

        timing_log = (
            f"{timestamp} - Pipeline: {self.config.pipeline_name}, "
            f"{mode_info}"
            f"Hierarchical Processing Time: {execution_time:.2f} seconds\n"
        )

    # def _log_execution_time(self, execution_time: float) -> None:
    #     """
    #     Log execution time with hierarchical processing details to the configured analysis directory.
    #
    #     This method creates a persistent record of processing times, which can be useful
    #     for performance monitoring and optimization.
    #     """
    #     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #     timing_log = (
    #         f"{timestamp} - Pipeline: {self.config.pipeline_name}, "
    #         f"Mode: {self.config.mode}, "
    #         f"Hierarchical Processing Time: {execution_time:.2f} seconds\n"
    #     )
    #
    #     timing_log_path = self.config.paths_config.analysis_dir / "node_creation_times.txt"
    #     with open(timing_log_path, 'a') as f:
    #         f.write(timing_log)

    """
    NEW (Metadata Extraction): New methods
    """

    def _batch_model_descriptions(self, nodes: list[BaseNode]) -> list[list[BaseNode]]:
        """
        Group similar model descriptions together for batch processing.
        Uses token limits and batch settings from metadata extraction configuration.
        """
        # Get token settings from config
        max_tokens = self.config.metadata_extraction.max_tokens_per_batch
        prompt_tokens = self.config.metadata_extraction.tokens_for_prompt
        response_tokens = self.config.metadata_extraction.tokens_for_response
        available_tokens = max_tokens - prompt_tokens - response_tokens

        batches = []
        current_batch = []
        current_token_count = 0

        for node in nodes:
            # Rough approximation: 4 chars = 1 token
            estimated_tokens = len(node.text) // 4

            if current_token_count + estimated_tokens > available_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_token_count = 0

            current_batch.append(node)
            current_token_count += estimated_tokens

        # Add any remaining nodes
        if current_batch:
            batches.append(current_batch)

        # Log batch statistics if verbose mode is enabled
        if self.config.verbose:
            self.logger.info(f"Created {len(batches)} batches from {len(nodes)} nodes")
            for i, batch in enumerate(batches):
                self.logger.debug(f"Batch {i + 1}: {len(batch)} nodes, ~{current_token_count} tokens")

        return batches

    def _extract_document_metadata(self, doc_content: str) -> dict[str, any]:
        """
        Extract document-level metadata using LLM.
        Uses prompt template and settings from metadata extraction configuration.
        """
        try:
            client = self.config.metadata_extraction.get_client()
            message = client.messages.create(
                messages=[{
                    "role": "user",
                    "content": self.config.metadata_extraction.document_level_prompt.format(
                        text=doc_content
                    )
                }],
                model=self.config.metadata_extraction.model_name, #changed by me from vendor_multimodal_model_name
                max_tokens=1000,
                temperature=self.config.metadata_extraction.temperature,
            )

            if message.content:
                text_content = (
                    message.content[0].text
                    if isinstance(message.content, list) and message.content
                    else str(message.content)
                )

                try:
                    doc_metadata = json.loads(text_content)
                    if self.config.verbose:
                        self.logger.debug(
                            f"Successfully parsed document metadata: {json.dumps(doc_metadata, indent=2)}")
                    return doc_metadata
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse document metadata: {str(e)}")
                    return self._get_default_doc_metadata()

            return self._get_default_doc_metadata()

        except Exception as e:
            self.logger.error(f"Document metadata extraction failed: {str(e)}")
            return self._get_default_doc_metadata()

    def _get_default_doc_metadata(self) -> dict:
        """Helper method to return default document metadata structure."""
        return {
            "manufacturer": None,
            "document_type": None,
            "year_published": None,
            "equipment_categories": [],
            "models_included": []
        }

    def _extract_batch_metadata(self, batch: list[BaseNode]) -> dict[str, dict]:
        """
        Extract metadata for a batch of model descriptions using the configured LLM.

        This method constructs a single text from all nodes in the batch, sends it to the
        LLM for metadata extraction, and returns the matched metadata for each node.
        """
        # Combine all node texts with clear separation
        batch_text = "\n\n".join(node.text for node in batch)

        try:
            client = self.config.metadata_extraction.get_client()
            message = client.messages.create(
                messages=[{
                    "role": "user",
                    "content": self.config.metadata_extraction.model_batch_prompt.format(
                        text=batch_text
                    )
                }],
                model=self.config.metadata_extraction.model_name,
                max_tokens=2000,
                temperature=self.config.metadata_extraction.temperature,
            )

            if message.content:
                text_content = (
                    message.content[0].text
                    if isinstance(message.content, list) and message.content
                    else str(message.content)
                )

                if self.config.verbose:
                    self.logger.debug(f"Raw LLM response text: {text_content}")

                try:
                    metadata_list = json.loads(text_content)
                    if self.config.verbose:
                        self.logger.debug(f"Successfully parsed metadata: {json.dumps(metadata_list, indent=2)}")
                    return self._match_metadata_to_nodes(batch, metadata_list)

                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON parsing failed: {str(e)}")
                    return {}

            return {}

        except Exception as e:
            self.logger.error(f"Batch metadata extraction failed: {str(e)}")
            if self.config.verbose:
                self.logger.error(f"Full error: {str(e.__class__.__name__)}")
                import traceback
                self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return {}

    def _match_metadata_to_nodes(
            self,
            nodes: list[BaseNode],
            metadata_list: list[dict[str, any]]
    ) -> dict[str, dict]:
        """
        Match extracted metadata to nodes based on model numbers and headers.
        Uses matching threshold from metadata extraction configuration.
        """
        matched_metadata = {}
        threshold = self.config.metadata_extraction.metadata_matching_threshold

        for node in nodes:
            node_text = node.text.lower()
            header_text = node.metadata.get("header_info", {}).get("text", "").lower()

            if self.config.verbose:
                self.logger.debug(f"\nAttempting to match node: {node.node_id}")
                self.logger.debug(f"Header text: {header_text}")

            best_match = None
            best_score = 0

            for metadata in metadata_list:
                model_num = metadata.get("model_number", "").lower()
                if not model_num:
                    continue

                header_exact_match = 1.0 if model_num in header_text else 0.0
                text_match = 1.0 if model_num in node_text else 0.0
                score = max(header_exact_match, text_match)

                if score > best_score:
                    best_score = score
                    best_match = metadata

            if best_match and best_score > threshold:
                matched_metadata[node.node_id] = best_match
                if self.config.verbose:
                    self.logger.debug(f"Matched node {node.node_id} with score {best_score}")
            else:
                if self.config.verbose:
                    self.logger.debug(f"No strong match found for node {node.node_id}")

        return matched_metadata

    def _filter_nodes_by_pages(self, nodes: list[BaseNode], pages: list[int] | None) -> list[BaseNode]:
        """
        Filter nodes based on specified page numbers.

        Args:
            nodes: List of nodes to filter
            pages: List of page numbers to include, or None for all pages

        Returns:
            Filtered list of nodes
        """
        if pages is None:
            return nodes

        return [
            node for node in nodes
            if node.metadata["document_info"]["page_num"] in pages
        ]

    """
    Test Method, may be deleted also
    """

    def _enhance_nodes_with_metadata(
            self,
            nodes: list[BaseNode]
    ) -> None:
        """
        Enhance nodes with both document and equipment metadata.

        Args:
            nodes: List of nodes to enhance with metadata
        """
        if not nodes:
            self.logger.info("No nodes provided for metadata enhancement")
            return

        self.logger.info(f"Enhancing metadata for {len(nodes)} nodes")
        page_nums = sorted(set(node.metadata["document_info"]["page_num"] for node in nodes))
        self.logger.info(f"Processing pages: {page_nums}")

        try:
            # First, extract document-level metadata from all provided nodes
            doc_text = "\n\n".join(node.text for node in nodes)
            doc_metadata = self._extract_document_metadata(doc_text)
            self.logger.debug(f"Extracted document metadata: {json.dumps(doc_metadata, indent=2)}")

            # Process nodes in batches for equipment metadata
            model_metadata = {}
            for batch in self._batch_model_descriptions(nodes):
                self.logger.info(f"Processing batch of {len(batch)} nodes")
                self.logger.info("Batch content:")
                for node in batch:
                    page_num = node.metadata["document_info"]["page_num"]
                    self.logger.info(f"Node {node.node_id} (Page {page_num}): {node.metadata['header_info']['text']}")

                raw_batch_metadata = self._extract_batch_metadata(batch)
                if raw_batch_metadata:
                    model_metadata.update(raw_batch_metadata)
                    self.logger.info(f"Updated model metadata. Current count: {len(model_metadata)}")
                else:
                    self.logger.warning("No metadata extracted from this batch")

            self.logger.debug(f"Final model metadata: {json.dumps(model_metadata, indent=2)}")

            # Initialize standard metadata structure for all nodes
            timestamp = datetime.now().isoformat()
            metadata_enhanced_count = 0

            for node in nodes:
                # Ensure all required metadata fields exist
                if "document_metadata" not in node.metadata:
                    node.metadata["document_metadata"] = {
                        "manufacturer": None,
                        "document_type": None,
                        "year_published": None,
                        "equipment_categories": [],
                        "models_included": []
                    }

                if "equipment_metadata" not in node.metadata:
                    node.metadata["equipment_metadata"] = {
                        "product_name": None,
                        "model_number": None,
                        "manufacturer": None,
                        "category": None,
                        "subcategory": None,
                        "year": None,
                        "specifications": [],
                        "capabilities": [],
                        "content_types": []
                    }

                if "extraction_info" not in node.metadata:
                    node.metadata["extraction_info"] = {
                        "extraction_model": None,
                        "metadata_version": None,
                        "extraction_timestamp": None
                    }

                # Update document metadata
                node.metadata["document_metadata"] = doc_metadata
                node.metadata["extraction_info"]["extraction_model"] = self.config.metadata_extraction.model_name
                node.metadata["extraction_info"]["metadata_version"] = "1.0"
                node.metadata["extraction_info"]["extraction_timestamp"] = timestamp

                # Add equipment metadata if available
                if node.node_id in model_metadata:
                    node.metadata["equipment_metadata"] = model_metadata[node.node_id]
                    self.logger.info(
                        f"Enhanced node {node.node_id} (Page {node.metadata['document_info']['page_num']}) with metadata"
                    )
                    metadata_enhanced_count += 1
                else:
                    self.logger.warning(
                        f"No metadata found for node {node.node_id} (Page {node.metadata['document_info']['page_num']})"
                    )

            self.logger.info(
                f"Enhanced {metadata_enhanced_count} out of {len(nodes)} nodes with metadata"
            )

        except Exception as e:
            self.logger.error(f"Metadata enhancement failed: {str(e)}")
            self.logger.error(f"Full error: {str(e.__class__.__name__)}")
            import traceback
            self.logger.error(f"Stack trace: {traceback.format_exc()}")