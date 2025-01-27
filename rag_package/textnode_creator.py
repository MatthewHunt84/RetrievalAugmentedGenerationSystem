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
import pickle
from llama_index.core.schema import BaseNode
import hashlib
import llama_index
from typing import Optional

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
        self._setup_logging()

        self.logger.info("Initializing TextNodeCreator")

        # Initialize paths from config
        self.parsed_results_path = self.config.paths_config.parsed_results_path
        self.output_dir = self.config.paths_config.output_dir
        self.analysis_dir = self.config.paths_config.analysis_dir

        # Now that we have logging and paths, create the cache instance
        # Pass both the creator (self) and the logger
        self.cache = self.NodeCache(self, self.logger)

        # Initialize the cache paths now that everything is ready
        self.cache.initialize_paths()

        # Continue with the rest of your initialization
        self._ensure_output_paths()
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
        If no pages are specified for metadata extraction, processes all pages.

        Args:
            metadata_extraction_test_pages: List of page numbers to enhance with metadata.
                If None, will use test_pages from config.
                If config test_pages is also None, will process all pages.
            node_analysis_pages: List of page numbers to include in analysis output.
                If None, will use the same pages as metadata_extraction_test_pages
        """
        start_time = time.time()

        try:
            self.logger.info("Beginning hierarchical node creation process...")

            # Create all nodes first
            all_nodes = self._create_base_nodes()

            # Use config test pages if no specific pages provided, otherwise use all pages
            metadata_pages = (
                metadata_extraction_test_pages if metadata_extraction_test_pages is not None
                else (
                    self.config.test_pages if hasattr(self.config, 'test_pages') and self.config.test_pages is not None
                    else None)  # None here will cause _filter_nodes_by_pages to return all nodes
            )

            # For metadata extraction - now will process all pages if metadata_pages is None
            self.logger.info(
                "Extracting metadata for all pages" if metadata_pages is None
                else f"Extracting metadata for pages: {metadata_pages}"
            )
            metadata_nodes = self._filter_nodes_by_pages(all_nodes, metadata_pages)
            if metadata_nodes:
                self._enhance_nodes_with_metadata(metadata_nodes)

            # For analysis output - use metadata pages if no specific analysis pages given
            analysis_pages = node_analysis_pages if node_analysis_pages is not None else metadata_pages
            if analysis_pages is not None:  # Only filter for analysis if specific pages requested
                self.logger.info(f"Analyzing nodes from pages: {analysis_pages}")
                analysis_nodes = self._filter_nodes_by_pages(all_nodes, analysis_pages)
                if analysis_nodes:
                    self.analyze_node_hierarchy(analysis_nodes)

            execution_time = time.time() - start_time
            self._log_execution_time(execution_time)

            return all_nodes

        except Exception as e:
            self.logger.error(f"Hierarchical node creation failed: {str(e)}")
            raise TextNodeCreationError(f"Failed to create nodes: {str(e)}")

    def _create_base_nodes(self) -> list[BaseNode]:
        """Create the initial set of nodes from the parsed content."""
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

                nodes = self._process_document_content(
                    page_dict["md"],
                    base_metadata
                )
                all_nodes.extend(nodes)
                self.logger.info(
                    f"Created {len(nodes)} nodes for page {idx + 1}"
                )

        return all_nodes


    def analyze_node_hierarchy(self, nodes: list[BaseNode]) -> None:
        """
        Analyze the hierarchical structure of nodes, focusing on nodes that underwent
        metadata extraction. This provides visibility into how metadata decoration
        affected specific nodes in our document.

        The analysis creates a detailed report showing the structure, relationships,
        and metadata of nodes from the pages we're processing. This helps us verify
        that our metadata extraction and decoration is working as expected.

        Args:
            nodes: List of nodes that were processed for metadata extraction
        """
        analysis_path = self.analysis_dir / f"{self.config.pipeline_name}_raw_node_analysis.txt"

        self.logger.info(f"Starting node hierarchy analysis for {len(nodes)} nodes")
        self.logger.info(f"Writing analysis to {analysis_path}")

        with open(analysis_path, 'w', encoding='utf-8') as f:
            # Write LLM calls summary
            f.write("=== LLM Calls Analysis ===\n")
            f.write(f"Document Metadata Extraction Calls: 1\n")
            num_batches = (
                                      len(nodes) + self.config.metadata_extraction.batch_size - 1) // self.config.metadata_extraction.batch_size
            f.write(f"Batch Metadata Extraction Calls: {num_batches}\n")
            f.write(f"Total LLM Calls: {1 + num_batches}\n\n")

            # Get the unique pages we're analyzing
            pages = sorted(set(node.metadata["document_info"]["page_num"] for node in nodes))
            f.write(f"Analyzing nodes from pages: {pages}\n\n")

            # Analyze each level
            for level in [0, 1, 2]:
                f.write(f"\n=== Level {level} Nodes ===\n\n")

                # Filter nodes for current level
                level_nodes = [
                    node for node in nodes
                    if node.metadata["hierarchy_info"]["level"] == level
                ]

                if not level_nodes:
                    f.write(f"No level {level} nodes found in the analyzed pages.\n")
                else:
                    # Sort nodes by page number and position for clearer output
                    level_nodes.sort(
                        key=lambda n: (
                            n.metadata["document_info"]["page_num"],
                            n.text.find(n.metadata["header_info"]["text"])
                        )
                    )

                    for i, node in enumerate(level_nodes, 1):
                        page_num = node.metadata["document_info"]["page_num"]
                        f.write(f"Node {i} (Page {page_num}):\n")
                        f.write("=" * 50 + "\n")

                        # Display node attributes
                        f.write(f"Node ID: {node.node_id}\n")
                        f.write(f"Node Type: {node.__class__.__name__}\n")
                        f.write(f"Page Number: {page_num}\n")

                        # Display text content
                        f.write("\nText Content:\n")
                        f.write("-" * 20 + "\n")
                        f.write(f"{node.text}\n")

                        # Display relationships
                        f.write("\nRelationships:\n")
                        f.write("-" * 20 + "\n")
                        for rel_type, rel_ids in node.relationships.items():
                            f.write(f"{rel_type}: {sorted(list(rel_ids))}\n")

                        # Display metadata values with special attention to equipment metadata
                        f.write("\nMetadata Values:\n")
                        f.write("-" * 20 + "\n")
                        for key, value in node.metadata.items():
                            f.write(f"{key}:\n")
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    f.write(f"  {subkey}: {subvalue}\n")
                            else:
                                f.write(f"  {value}\n")

                            # Highlight if equipment metadata was successfully extracted
                            if key == "equipment_metadata" and any(value.values()):
                                f.write("  ** This node has equipment metadata **\n")

                        f.write("\n" + "=" * 50 + "\n\n")

                    # Summary statistics for this level
                    f.write(f"\nSummary for level {level}:\n")
                    f.write(f"Total nodes at level {level}: {len(level_nodes)}\n")
                    f.write(
                        f"Nodes with equipment metadata: {sum(1 for n in level_nodes if any(n.metadata.get('equipment_metadata', {}).values()))}\n")
                    f.write(
                        f"Nodes with document metadata: {sum(1 for n in level_nodes if any(n.metadata.get('document_metadata', {}).values()))}\n")
                    f.write("\n" + "=" * 50 + "\n\n")


    def _setup_logging(self):
        """
        Configure logging with proper level setting for both logger and handlers.
        """
        # Clear any existing handlers
        self.logger.handlers = []

        # Set the logger's level first
        self.logger.setLevel(self.config.logging_config.level)

        if self.config.logging_config.log_to_file:
            log_file = self.config.paths_config.analysis_dir / "node_creation.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.config.logging_config.level)
            file_handler.setFormatter(
                logging.Formatter(self.config.logging_config.format)
            )
            self.logger.addHandler(file_handler)

        if self.config.logging_config.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.config.logging_config.level)
            console_handler.setFormatter(
                logging.Formatter(self.config.logging_config.format)
            )
            self.logger.addHandler(console_handler)

        # Verify configuration
        if self.config.verbose:
            self.logger.info(f"Logger level set to: {self.logger.level}")


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
        try:
            self.logger.info("Starting document metadata extraction...")

            client = self.config.metadata_extraction.get_client()
            response = client.create_message(
                prompt=self.config.metadata_extraction.document_level_prompt.format(
                    text=doc_content
                ),
                max_tokens=1000
            )

            self._inspect_llm_response(response, "Document Metadata Extraction")

            if response["content"]:  # Access content from our standardized response format
                json_str, error = self._extract_json_from_text(response["content"], is_array=False)
                if json_str:
                    doc_metadata = json.loads(json_str)
                    self.logger.info("Successfully extracted document metadata")
                    if self.config.verbose:
                        self.logger.info(f"Document metadata content: {json.dumps(doc_metadata, indent=2)}")
                    return doc_metadata
                else:
                    self.logger.error(f"Failed to extract document metadata: {error}")
                    return self._get_default_doc_metadata()

            self.logger.info("No content in LLM response, using default metadata")
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

        This method processes multiple nodes at once to extract equipment metadata,
        using the standardized interface provided by our LLM clients.
        """
        batch_text = "\n\n".join(node.text for node in batch)

        try:
            self.logger.info(f"Processing batch of {len(batch)} nodes for metadata extraction")

            # Get the LLM client from the config
            client = self.config.metadata_extraction.get_client()

            # Use the standardized create_message interface instead of direct API calls
            response = client.create_message(
                prompt=self.config.metadata_extraction.model_batch_prompt.format(
                    text=batch_text
                ),
                max_tokens=2000
            )

            self._inspect_llm_response(response, "Batch Metadata Extraction")

            if response["content"]:  # Note: we now access content from our standardized response format
                # Extract JSON using our helper method
                json_str, error = self._extract_json_from_text(response["content"], is_array=True)

                if json_str:
                    metadata_list = json.loads(json_str)
                    self.logger.info(f"Successfully extracted metadata for batch")
                    if self.config.verbose:
                        self.logger.info(f"Processing metadata for {len(metadata_list)} models")
                    return self._match_metadata_to_nodes(batch, metadata_list)
                else:
                    self.logger.error(f"Failed to extract batch metadata: {error}")
                    return {}

            self.logger.info("No content in LLM response")
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
        Enhanced matching logic that considers various content patterns.
        First, we gather a batch of nodes that contain text about equipment (like product descriptions, specifications, features, etc.)
        We send all this text to the LLM with a prompt like "Extract metadata for any equipment models you find in this text."
        The LLM then returns a JSON array of metadata objects, each representing a different piece of equipment it found.
        Then comes the tricky part - we need to figure out which metadata object belongs to which node.
        We currently do this by looking for matching text patterns (like model numbers or product names) because the LLM doesn't know anything about our node IDs.
        It's just seeing text and extracting information from it.
        This is a better approach compared to passing up the node IDs with the batches, because often metadata spans multiple nodes and this matching approach is more likely to put things in the right place.
        """

        matched_metadata = {}
        threshold = self.config.metadata_extraction.metadata_matching_threshold

        for node in nodes:
            node_text = node.text.lower()
            header_text = node.metadata.get("header_info", {}).get("text", "").lower()
            header_level = node.metadata.get("header_info", {}).get("level", 0)

            if self.config.verbose:
                self.logger.info(f"\nEvaluating node: {node.node_id}")
                self.logger.info(f"Header text: {header_text}")
                self.logger.info(f"Header level: {header_level}")

            # Check hierarchy level from hierarchy_info (0 = top level)
            hierarchy_level = node.metadata.get("hierarchy_info", {}).get("level", 0)

            # Check if this is a top-level category/section header
            if hierarchy_level == 0 and any(keyword in header_text for keyword in
                                            ["overview", "section", "category", "products", "trenchers", "equipment"]):
                self.logger.info(
                    f"Node {node.node_id} is a top-level category header "
                    f"({header_text}) at hierarchy level {hierarchy_level}. "
                    f"Metadata extraction not applicable for structural nodes."
                )
                continue

            best_match = None
            best_score = 0

            for metadata in metadata_list:
                # Multiple matching criteria
                model_num = metadata.get("model_number", "").lower()
                product_name = metadata.get("product_name", "").lower()

                # Calculate various match scores
                model_in_header = float(model_num in header_text) if model_num else 0.0
                model_in_text = float(model_num in node_text) if model_num else 0.0
                product_in_header = float(product_name in header_text) if product_name else 0.0
                product_in_text = float(product_name in node_text) if product_name else 0.0

                # Additional matching for specification sections
                spec_match = 0.0
                if any(keyword in header_text for keyword in ["specifications", "features", "details"]):
                    spec_match = float(model_num in node_text or product_name in node_text)

                # Combined score with weights
                score = max(
                    model_in_header * 1.0,  # Highest priority
                    model_in_text * 0.8,
                    product_in_header * 0.7,
                    product_in_text * 0.6,
                    spec_match * 0.5
                )

                if score > best_score:
                    best_score = score
                    best_match = metadata

            if best_match and best_score > threshold:
                matched_metadata[node.node_id] = best_match
                if self.config.verbose:
                    self.logger.info(
                        f"Successfully matched node {node.node_id} "
                        f"to model {best_match.get('model_number')} "
                        f"with confidence score {best_score:.2f}"
                    )
            else:
                if hierarchy_level > 0:
                    # This is a content node that we expected to match but couldn't
                    self.logger.warning(
                        f"Unable to find matching metadata for content node {node.node_id} "
                        f"(Page {node.metadata['document_info']['page_num']}, "
                        f"Header: {header_text}, Hierarchy Level: {hierarchy_level})"
                    )
                else:
                    # This is a structural node (header, category, etc.)
                    self.logger.debug(
                        f"No metadata match needed for structural node {node.node_id} "
                        f"(Page {node.metadata['document_info']['page_num']}, "
                        f"Header: {header_text}, Hierarchy Level: {hierarchy_level})"
                    )

        return matched_metadata

    def _filter_nodes_by_pages(self, nodes: list[BaseNode], pages: list[int] | None) -> list[BaseNode]:
        """
        Filter nodes based on specified page numbers.
        If pages is None, returns all nodes.
        """
        if pages is None:
            self.logger.info("No pages specified for filtering, returning all nodes")
            return nodes

        filtered_nodes = [
            node for node in nodes
            if node.metadata["document_info"]["page_num"] in pages
        ]

        self.logger.info(f"Filtered {len(nodes)} nodes to {len(filtered_nodes)} nodes from pages {pages}")
        return filtered_nodes


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

    def _extract_json_from_text(self, text_content: str, is_array: bool = False) -> tuple[str | None, str | None]:
        """
        Extract JSON content from text that may contain additional context or formatting.
        Sometimes the LLM likes to add an introductory sentence like "Here is the JSON you requested:" before the valid JSON
        This extracts relevant JSON only

        Args:
            text_content: The full text response that contains JSON
            is_array: Whether to look for array brackets ([]) or object brackets ({})

        Returns:
            tuple: (extracted JSON string or None, error message or None)
            If successful, returns (json_str, None)
            If unsuccessful, returns (None, error_message)
        """
        try:
            # Determine which brackets to look for based on expected JSON type
            start_char = '[' if is_array else '{'
            end_char = ']' if is_array else '}'

            # Find the JSON boundaries
            json_start = text_content.find(start_char)
            json_end = text_content.rfind(end_char)

            if json_start != -1 and json_end != -1:
                json_str = text_content[json_start:json_end + 1]

                # Validate that we can parse it
                json.loads(json_str)  # This will raise JSONDecodeError if invalid
                return json_str, None

            return None, f"Could not find matching {start_char}{end_char} in response"

        except json.JSONDecodeError as e:
            return None, f"Invalid JSON structure: {str(e)}"

    def _inspect_llm_response(self, response, context: str) -> None:
        """
        Inspects LLM responses with support for both Anthropic and OpenAI formats.

        This method provides detailed inspection of LLM responses regardless of their source.
        It intelligently detects the response format and extracts relevant information
        in a consistent way for logging and debugging purposes.

        Args:
            response: The raw response from either Anthropic or OpenAI
            context: A string describing where this inspection is happening
        """
        if self.config.verbose:
            self.logger.info(f"\n=== LLM Response Overview ({context}) ===")

            # First, let's identify the response type
            response_type = type(response).__name__
            self.logger.info(f"Response type: {response_type}")

            # Extract content based on the response type
            content = None
            api_source = None

            if hasattr(response, 'choices'):
                # This is likely an OpenAI response
                api_source = 'OpenAI'
                if response.choices:
                    content = response.choices[0].message.content
                    self.logger.info(f"Model used: {getattr(response, 'model', 'unknown')}")
                    self.logger.info(f"Total tokens: {getattr(response, 'usage', {}).get('total_tokens', 'unknown')}")

            elif hasattr(response, 'content'):
                # This is likely an Anthropic response
                api_source = 'Anthropic'
                if isinstance(response.content, list):
                    # Handle Anthropic's list-style content
                    content = response.content[0].text if response.content else None
                else:
                    # Handle Anthropic's string content
                    content = str(response.content)

                self.logger.info(f"Model used: {getattr(response, 'model', 'unknown')}")

            self.logger.info(f"API Source: {api_source or 'Unknown'}")

            # Process the extracted content consistently, regardless of source
            if content:
                self.logger.info(f"Content type: {type(content)}")
                self.logger.info(f"Content length: {len(content)}")

                # Provide a preview of the content
                preview_length = 500
                content_preview = content[:preview_length]
                if len(content) > preview_length:
                    content_preview += "..."
                self.logger.info(f"Content preview:\n{content_preview}")

                # Check if content looks like JSON
                try:
                    json.loads(content)
                    self.logger.info("Content appears to be valid JSON")
                except json.JSONDecodeError as e:
                    self.logger.info(f"Content is not valid JSON: {str(e)}")

                    # Look for JSON-like structures that might need extraction
                    if '{' in content and '}' in content:
                        self.logger.info("Content contains JSON-like structures that may need extraction")
            else:
                self.logger.info("No content found in response")

            # Log any additional metadata that might be helpful
            if hasattr(response, 'created'):
                self.logger.info(f"Response timestamp: {response.created}")

            self.logger.info("=== End Response Overview ===\n")

    class NodeCache:
        def __init__(self, creator: 'TextNodeCreator', logger: logging.Logger):
            """
            Initialize the cache manager with the parent TextNodeCreator and a logger.

            Args:
                creator: The parent TextNodeCreator instance
                logger: A configured logger instance
            """
            self.creator = creator
            self.logger = logger

            # We'll set up the paths later
            self.cache_dir = None
            self.metadata_path = None
            self.nodes_path = None

        def initialize_paths(self):
            """Initialize cache paths after TextNodeCreator is fully set up."""
            self.cache_dir = self.creator.output_dir
            self.metadata_path = self.cache_dir / "node_cache_metadata.pkl"
            self.nodes_path = self.cache_dir / "cached_nodes.pkl"

        def _get_parsed_results_hash(self) -> str:
            """
            Computes a hash of the parsed results file to detect content changes.
            This hash is used to invalidate the cache when the source data changes.

            Returns:
                str: A hex string representing the SHA-256 hash of the file contents,
                     or an empty string if the file doesn't exist.
            """
            parsed_results_path = self.creator.parsed_results_path
            if not parsed_results_path.exists():
                self.logger.debug(f"Parsed results file not found at: {parsed_results_path}")
                return ""

            try:
                with open(parsed_results_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    self.logger.debug(f"Computed hash for parsed results: {file_hash[:8]}...")
                    return file_hash
            except Exception as e:
                self.logger.error(f"Failed to compute hash for parsed results: {str(e)}")
                return ""

        def _compute_config_hash(self) -> str:
            """
            Creates a deterministic hash of the configuration to detect changes.
            Uses all relevant fields from NodeCreationConfig that could affect node creation.
            """
            config_dict = {
                "pipeline_name": self.creator.config.pipeline_name,
                "chunk_sizes": self.creator.config.chunk_sizes,
                "chunk_overlap": self.creator.config.chunk_overlap,
                "test_pages": self.creator.config.test_pages,
                "hierarchical_config": self.creator.config.hierarchical_config.__dict__,
                "metadata_extraction": {
                    "model": self.creator.config.metadata_extraction.model_name,
                    "batch_size": self.creator.config.metadata_extraction.batch_size,
                    "max_tokens": self.creator.config.metadata_extraction.max_tokens_per_batch
                },
                "base_instruction": self.creator.config.base_instruction
            }

            config_str = json.dumps(config_dict, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()

        def save_nodes(self, nodes: list[BaseNode]) -> None:
            """
            Saves nodes and their metadata to cache.
            """
            try:
                metadata = {
                    "creation_timestamp": datetime.now().isoformat(),
                    "pipeline_name": self.creator.config.pipeline_name,
                    "config_hash": self._compute_config_hash(),
                    "parsed_results_hash": self._get_parsed_results_hash(),
                    "llama_index_version": llama_index.core.__version__,
                    "node_count": len(nodes)
                }

                self.logger.info(f"Saving {len(nodes)} nodes to cache...")

                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)

                with open(self.nodes_path, 'wb') as f:
                    pickle.dump(nodes, f)

                self.logger.info(f"Successfully cached nodes to {self.cache_dir}")
                self.logger.debug(f"Cache metadata: {metadata}")

            except Exception as e:
                self.logger.error(f"Failed to save nodes to cache: {str(e)}")
                raise

        def load_nodes(self) -> Optional[list[BaseNode]]:
            """
            Attempts to load cached nodes, performing validation checks.
            """
            try:
                if self.creator.config.invalidate_cache:
                    self.logger.info("Cache invalidation requested in config")
                    return None

                if not (self.metadata_path.exists() and self.nodes_path.exists()):
                    self.logger.info("No cached nodes found")
                    return None

                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)

                # Validate cache
                current_config_hash = self._compute_config_hash()
                if metadata["config_hash"] != current_config_hash:
                    self.logger.info("Configuration has changed - invalidating cache")
                    return None

                current_results_hash = self._get_parsed_results_hash()
                if metadata["parsed_results_hash"] != current_results_hash:
                    self.logger.info("Parsed results have changed - invalidating cache")
                    return None

                with open(self.nodes_path, 'rb') as f:
                    nodes = pickle.load(f)

                self.logger.info(f"Successfully loaded {len(nodes)} nodes from cache")
                self.logger.debug(f"Cache metadata: {metadata}")

                return nodes

            except Exception as e:
                self.logger.error(f"Error loading cached nodes: {str(e)}")
                return None

        def clear(self) -> None:
            """Removes all cached data."""
            try:
                if self.metadata_path.exists():
                    self.metadata_path.unlink()
                if self.nodes_path.exists():
                    self.nodes_path.unlink()
                self.logger.info("Cache cleared successfully")
            except Exception as e:
                self.logger.error(f"Error clearing cache: {str(e)}")
                raise

    def create_or_load_nodes(
            self,
            metadata_extraction_test_pages: list[int] = None,
            node_analysis_pages: list[int] = None,
            force_refresh: bool = False
    ) -> list[BaseNode]:
        """
        Main entry point for node creation that implements caching.
        This method either loads nodes from cache or creates new ones as needed.

        Args:
            metadata_extraction_test_pages: Optional list of pages to enhance with metadata
            node_analysis_pages: Optional list of pages to include in analysis output
            force_refresh: If True, bypasses cache and creates new nodes

        Returns:
            List of BaseNode objects
        """
        if force_refresh or self.config.do_not_cache:
            self.logger.info("Bypassing cache due to force_refresh or do_not_cache setting")
            return self.create_nodes(
                metadata_extraction_test_pages=metadata_extraction_test_pages,
                node_analysis_pages=node_analysis_pages
            )

        # Try loading from cache first
        cached_nodes = self.cache.load_nodes()
        if cached_nodes is not None:
            return cached_nodes

        # Create new nodes if cache miss
        self.logger.info("Creating new nodes...")
        nodes = self.create_nodes(
            metadata_extraction_test_pages=metadata_extraction_test_pages,
            node_analysis_pages=node_analysis_pages
        )

        # Cache the newly created nodes if caching is enabled
        if not self.config.do_not_cache:
            self.cache.save_nodes(nodes)

        return nodes
