from pathlib import Path
import json
import re
import pickle
import time
from datetime import datetime
import logging
from llama_index.core.schema import TextNode, Document
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
from rag_package.errors import TextNodeCreationError
from rag_package.rag_config import (
    NodeCreationConfig,
    EquipmentSpecifications,
    Measurement
)


class TextNodeCreator:
    """
    Enhanced text node creator that combines hierarchical parsing with detailed
    technical specification extraction. This class maintains document structure while
    adding rich technical metadata extraction capabilities.
    """

    def __init__(self, node_config: NodeCreationConfig):
        """
        Initialize the creator with enhanced configuration settings while maintaining
        hierarchical parsing capabilities.
        """
        self.config = node_config
        self.parsed_results_path = Path(node_config.parsed_results_path)
        self.output_dir = Path(node_config.output_dir)
        self.analysis_dir = Path("analysis")

        # Ensure required directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging with both file and console handlers
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._setup_logging()

        try:
            print("Initializing hierarchical document processor...")
            self.hierarchical_parser = self._initialize_hierarchical_parser()
            print("Successfully initialized document processor")

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise TextNodeCreationError(f"Failed to initialize parser: {str(e)}")

    def _setup_logging(self):
        """
        Configure logging to write to both a file and the console.
        The file handler keeps a record of processing events, while the console
        handler provides immediate feedback during execution.
        """
        # Clear any existing handlers
        self.logger.handlers = []

        # Create file handler that logs even debug messages
        log_file = self.analysis_dir / "node_creation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _initialize_hierarchical_parser(self) -> HierarchicalNodeParser:
        """
        Initialize the hierarchical parser with appropriate chunk sizes for different
        header levels, maintaining document structure while allowing for detailed content analysis.

        The method converts our dictionary-based configuration into the list format
        expected by HierarchicalNodeParser. The conversion ensures that:
        1. Chunk sizes are ordered correctly by hierarchy level
        2. The parser receives properly formatted inputs
        3. Each level's chunking configuration is properly aligned
        """
        # Convert our dictionary-based chunk sizes into a list format
        # The list length should be max(level) + 1 since levels are 0-based in the parser
        max_level = max(self.config.hierarchy_config.chunk_sizes.keys())
        chunk_sizes = [256] * (max_level + 1)  # Default size for any unspecified levels

        # Fill in the specified chunk sizes
        for level, size in self.config.hierarchy_config.chunk_sizes.items():
            chunk_sizes[level] = size

        # Create sentence splitters for each level
        # We still use the original dictionary for our splitter map since it's more intuitive
        splitter_map = {
            f"level_{level}": SentenceSplitter(
                chunk_size=size,
                chunk_overlap=self.config.hierarchy_config.chunk_overlaps[level]
            )
            for level, size in self.config.hierarchy_config.chunk_sizes.items()
        }

        # Create parser IDs in descending order (highest to lowest level)
        parser_ids = [f"level_{level}"
                      for level in sorted(self.config.hierarchy_config.chunk_sizes.keys(),
                                          reverse=True)]

        # Initialize the hierarchical parser with the converted configuration
        return HierarchicalNodeParser(
            chunk_sizes=chunk_sizes,  # Now passing a list
            node_parser_ids=parser_ids,
            node_parser_map=splitter_map
        )

    def _extract_measurements_and_context(self, text: str) -> dict[str, list[tuple[float, str, str]]]:
        """
        Generic measurement extraction that captures numerical values with their units
        and surrounding context. Uses natural language processing patterns to understand
        the relationship between measurements and their descriptions.
        """
        # Pattern that catches any number followed by any unit
        measurement_pattern = r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*([a-zA-Z]+(?:/[a-zA-Z]+)?)'

        measurements = []
        for match in re.finditer(measurement_pattern, text):
            # Get the full sentence containing this measurement for context
            sentence_pattern = r'[^.!?]*{}[^.!?]*[.!?]'.format(re.escape(match.group(0)))
            context_match = re.search(sentence_pattern, text)
            context = context_match.group(0) if context_match else text[max(0, match.start() - 50):min(len(text),
                                                                                                       match.end() + 50)]

            value_text, unit = match.groups()

            # Handle range values
            if '-' in value_text:
                low, high = map(float, value_text.split('-'))
                value = (low + high) / 2  # Store average while keeping range in context
            else:
                value = float(value_text)

            measurements.append((value, unit, context.strip()))

        return measurements

    def _extract_measurements_and_context(self, text: str) -> dict[str, list[tuple[float, str, str]]]:
        """
        Enhanced measurement extraction that distinguishes between actual measurements
        and model identifiers/other numerical patterns.
        """
        # First, identify and mask model numbers to protect them from measurement extraction
        model_pattern = r'(?:[A-Z][a-zA-Z]*\s)?(\d{3,4}[A-Z]{2,3})'
        model_numbers = set(re.findall(model_pattern, text))
        protected_text = text

        # Create a mapping of protected strings
        protected_mappings = {}
        for i, model in enumerate(model_numbers):
            placeholder = f"__MODEL_{i}__"
            protected_mappings[placeholder] = model
            protected_text = protected_text.replace(model, placeholder)

        # Now look for measurements in the protected text
        # Define common measurement units
        valid_units = {
            'length': ['in', 'ft', 'm', 'mm', 'cm'],
            'weight': ['lb', 'kg', 'ton'],
            'power': ['hp', 'kw'],
            'pressure': ['psi', 'bar', 'kpa'],
            'flow': ['gpm', 'lpm'],
            'speed': ['rpm', 'mph', 'kph'],
            'temperature': ['f', 'c']
        }

        # Flatten the valid units for pattern matching
        all_valid_units = '|'.join([unit for units in valid_units.values() for unit in units])

        # Pattern that catches numbers followed by valid units
        measurement_pattern = f'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*({all_valid_units})'

        measurements = []
        for match in re.finditer(measurement_pattern, protected_text, re.IGNORECASE):
            value_text, unit = match.groups()

            # Get the surrounding context
            context_start = max(0, match.start() - 50)
            context_end = min(len(protected_text), match.end() + 50)
            context = protected_text[context_start:context_end]

            # Restore any model numbers in the context
            for placeholder, model in protected_mappings.items():
                context = context.replace(placeholder, model)

            # Process the value (handling ranges)
            if '-' in value_text:
                low, high = map(float, value_text.split('-'))
                value = (low + high) / 2  # Store average with range in context
            else:
                value = float(value_text)

            measurements.append((value, unit.lower(), context.strip()))

        return measurements

    def _identify_model_information(self, text: str) -> dict:
        """
        Separate method to handle model identification and related metadata.
        """
        model_patterns = {
            'full_model': r'(\d{3,4}[A-Z]{2,3}(?:\s+(?:Standard|Mini|Micro|Track)\s+Trencher)?)',
            'release_year': r'(?:introduced|released|developed)(?:\sin|\s)(?:the\syear\s)?(\d{4})',
            'category': r'(?:Standard|Mini|Micro|Track)\s+Trencher'
        }

        model_info = {}

        # Extract model information
        if full_model_match := re.search(model_patterns['full_model'], text):
            model_info['model_number'] = full_model_match.group(1)

        # Extract release year if present
        if year_match := re.search(model_patterns['release_year'], text):
            model_info['release_year'] = int(year_match.group(1))

        # Extract category if present
        if category_match := re.search(model_patterns['category'], text):
            model_info['category'] = category_match.group(0)

        return model_info

    def _infer_measurement_type(self, context: str, unit: str) -> str:
        """
        Infers the type of measurement based on context and unit.
        Uses a combination of unit analysis and contextual clues.
        """
        # First, try to understand the measurement from context
        context_lower = context.lower()

        # Look for descriptive phrases that explain what's being measured
        descriptor_matches = re.findall(r'(?:the|a|an)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+(?:is|of|at|was|measures)',
                                        context_lower)

        if descriptor_matches:
            return '_'.join(descriptor_matches[0].split())

        # If no clear context, create a generic but meaningful identifier
        return f"measurement_{unit}"

    def _extract_equipment_metadata(self, text: str) -> EquipmentSpecifications:
        """
        Enhanced metadata extraction that properly handles both measurements and model information.
        """
        specs = EquipmentSpecifications()

        # First, extract model information
        model_info = self._identify_model_information(text)
        specs.basic_info.update(model_info)

        # Then extract actual measurements
        measurements = self._extract_measurements_and_context(text)

        # Process each valid measurement
        for value, unit, context in measurements:
            measurement_type = self._infer_measurement_type(context, unit)
            specs.add_specification(
                name=measurement_type,
                value=value,
                unit=unit,
                context=context,
                confidence=0.9  # Higher confidence because we're using validated units
            )

        return specs

    def _infer_specification_name(self, context: str, unit: str) -> str:
        """
        Infer a meaningful name for a specification based on its context and unit.
        Uses common terminology patterns in technical documentation.
        """
        # Unit-based naming patterns
        unit_name_map = {
            'lb': 'weight',
            'kg': 'weight',
            'in': 'length',
            'ft': 'height',
            'hp': 'power',
            'kw': 'power_rating',
            'psi': 'pressure',
            'gpm': 'flow_rate'
        }

        # Look for specific measurement terms in context
        context_lower = context.lower()
        if 'depth' in context_lower:
            return 'working_depth'
        elif 'width' in context_lower:
            return 'width'
        elif 'height' in context_lower:
            return 'height'
        elif 'capacity' in context_lower:
            return 'capacity'

        # Default to unit-based naming if no specific context is found
        return unit_name_map.get(unit.lower(), f'measurement_{unit}')

    def _calculate_feature_confidence(self, feature: str, pattern: str) -> float:
        """
        Calculate confidence score for extracted features based on various heuristics.
        """
        confidence = 0.7  # Base confidence

        # Increase confidence for longer, more specific features
        if len(feature.split()) >= 3:
            confidence += 0.1

        # Increase confidence for features with technical terms
        technical_terms = {'hydraulic', 'automatic', 'electronic', 'integrated', 'system'}
        if any(term in feature.lower() for term in technical_terms):
            confidence += 0.1

        # Cap confidence at 0.95
        return min(0.95, confidence)

    def _process_content(self, content: str, base_metadata: dict) -> list[TextNode]:
        """
        Process content into nodes with enhanced metadata using the flexible
        EquipmentSpecifications structure.
        """
        try:
            sections = self._split_into_sections(content)
            enhanced_nodes = []

            for section in sections:
                header = section['header']
                text = section['content']
                full_text = f"{header}\n{text}" if header else text

                # Extract comprehensive equipment metadata
                equipment_specs = self._extract_equipment_metadata(full_text)

                # Determine hierarchy level
                hierarchy_level = self._detect_hierarchy_level(header if header else full_text)

                # Create enhanced metadata combining base metadata with equipment specifications
                enhanced_metadata = {
                    **base_metadata,
                    'equipment_specs': equipment_specs.to_dict(),
                    'hierarchy_level': hierarchy_level,
                    'section_header': header
                }

                # Create node with complete section
                enhanced_node = TextNode(
                    text=full_text,
                    metadata=enhanced_metadata
                )
                enhanced_nodes.append(enhanced_node)

            self.logger.debug(f"Processed {len(enhanced_nodes)} nodes from content")
            return enhanced_nodes

        except Exception as e:
            error_msg = f"Error processing content: {str(e)}"
            self.logger.error(error_msg)
            raise TextNodeCreationError(error_msg)

    def _split_into_sections(self, content: str) -> list[dict]:
        """
        Split content into logical sections based on headers.

        This method identifies section boundaries using Markdown headers and ensures
        that related content stays together. Each section includes its header and
        all content up to the next header of the same or higher level.

        Args:
            content: Raw markdown content

        Returns:
            list[dict]: List of sections, each with 'header' and 'content' keys
        """
        # Split content into lines for processing
        lines = content.strip().split('\n')
        sections = []
        current_section = {'header': '', 'content': []}

        for line in lines:
            # Check if line is a header
            if line.startswith('#'):
                # If we have content in the current section, save it
                if current_section['header'] or current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content']).strip()
                    sections.append(current_section)
                    current_section = {'header': '', 'content': []}

                # Start new section with this header
                current_section['header'] = line
            else:
                # Add non-header line to current section's content
                if line.strip():  # Only add non-empty lines
                    current_section['content'].append(line)

        # Add the last section if it has content
        if current_section['header'] or current_section['content']:
            current_section['content'] = '\n'.join(current_section['content']).strip()
            sections.append(current_section)

        return sections

    def _should_replace_existing_node(self, existing_node: TextNode, new_node: TextNode) -> bool:
        """
        Determine whether a new node should replace an existing node with the same content.

        This method implements the logic for choosing between duplicate nodes based on:
        1. Hierarchy level - higher levels are generally preferred
        2. Content completeness - longer content might be preferred
        3. Metadata richness - nodes with more detailed metadata might be preferred

        Returns:
            bool: True if the new node should replace the existing node
        """
        # Get hierarchy levels
        existing_level = existing_node.metadata.get('hierarchy_level', 0)
        new_level = new_node.metadata.get('hierarchy_level', 0)

        # If hierarchy levels are different, prefer the higher level
        if existing_level != new_level:
            return new_level > existing_level

        # If at same level, prefer the node with more complete content
        if len(new_node.text) != len(existing_node.text):
            return len(new_node.text) > len(existing_node.text)

        # If content length is the same, prefer the node with richer metadata
        existing_metadata_count = len(self._count_meaningful_metadata(existing_node.metadata))
        new_metadata_count = len(self._count_meaningful_metadata(new_node.metadata))

        return new_metadata_count > existing_metadata_count

    def _count_meaningful_metadata(self, metadata: dict) -> dict:
        """
        Count non-empty metadata fields, excluding basic fields like pipeline_name.
        """
        meaningful_fields = {}
        basic_fields = {'pipeline_name', 'hierarchy_config'}

        for key, value in metadata.items():
            if key not in basic_fields and value:
                if isinstance(value, dict):
                    if any(value.values()):
                        meaningful_fields[key] = value
                elif isinstance(value, (list, tuple)):
                    if value:
                        meaningful_fields[key] = value
                else:
                    meaningful_fields[key] = value

        return meaningful_fields

    def _detect_hierarchy_level(self, text: str) -> int:
        """
        Determine the hierarchy level of text using configured patterns.

        Args:
            text: The text content to analyze

        Returns:
            int: The detected hierarchy level (3 for highest, 1 for lowest, 0 for regular content)
        """
        return self.config.hierarchy_config.get_level_for_text(text)

    def create_nodes(self) -> list[TextNode]:
        """
        Create hierarchical nodes with timing measurement and logging.
        Returns a list of processed TextNodes.
        """
        start_time = time.time()

        try:
            self.logger.info("Beginning node creation process...")
            with open(self.parsed_results_path, 'r', encoding='utf-8') as f:
                md_json_objs = json.load(f)

            all_nodes = []
            for result in md_json_objs:
                document_name = Path(result["file_path"]).name
                self.logger.info(f"Processing document: {document_name}")

                for idx, page_dict in enumerate(result["pages"]):
                    metadata = {
                        **self.config.base_metadata,
                        "page_num": idx + 1,
                        "document_name": document_name,
                        "total_pages": len(result["pages"])
                    }

                    nodes = self._process_content(page_dict["md"], metadata)
                    all_nodes.extend(nodes)
                    self.logger.info(f"Created {len(nodes)} nodes for page {idx + 1}")

            if all_nodes:
                self._save_outputs(all_nodes)

            # Log execution time
            execution_time = time.time() - start_time
            self._log_execution_time(execution_time)

            return all_nodes

        except Exception as e:
            self.logger.error(f"Node creation failed: {str(e)}")
            raise TextNodeCreationError(f"Failed to create nodes: {str(e)}")

    def _save_outputs(self, nodes: list[TextNode]) -> None:
        """
        Save nodes and enhanced analysis information with the new metadata structure.
        """
        output_path = self.output_dir / f"{self.config.pipeline_name}_nodes.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(nodes, f)

        summary = {
            "total_nodes": len(nodes),
            "pipeline_config": self.config.base_metadata,
            "hierarchy_distribution": {
                level: len([n for n in nodes
                            if n.metadata.get('hierarchy_level') == level])
                for level in self.config.hierarchy_config.chunk_sizes.keys()
            },
            "metadata_statistics": {
                "average_specifications_per_node": sum(
                    len(n.metadata.get('equipment_specs', {}).get('numerical_specs', {}))
                    for n in nodes
                ) / len(nodes) if nodes else 0,
                "average_tags_per_node": sum(
                    len(n.metadata.get('equipment_specs', {}).get('attribute_tags', []))
                    for n in nodes
                ) / len(nodes) if nodes else 0,
            },
            "node_preview": [
                {
                    "text_preview": str(node.text),
                    "metadata": {
                        **node.metadata,
                        "equipment_specs": node.metadata.get('equipment_specs', {})
                    },
                    "node_id": node.node_id
                }
                for node in nodes[:5]
            ]
        }

        summary_path = self.output_dir / f"{self.config.pipeline_name}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Saved complete nodes to {output_path}")
        self.logger.info(f"Saved detailed summary to {summary_path}")

    def _log_execution_time(self, execution_time: float) -> None:
        """Log execution time to the analysis directory."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timing_log = f"{timestamp} - Pipeline: {self.config.pipeline_name}, Execution Time: {execution_time:.2f} seconds\n"

        with open(self.analysis_dir / "node_creation_times.txt", 'a') as f:
            f.write(timing_log)