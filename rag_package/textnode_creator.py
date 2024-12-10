from pathlib import Path
import json
import re
import pickle
import time
from datetime import datetime
import logging
from llama_index.core.schema import TextNode
from rag_package.errors import TextNodeCreationError
from rag_package.rag_config import NodeCreationConfig


class TextNodeCreator:
    """
    Text node creator that processes documents while maintaining their structure.
    """

    def __init__(self, node_config: NodeCreationConfig):
        """Initialize the creator with configuration settings."""
        self.config = node_config
        self.parsed_results_path = Path(node_config.parsed_results_path)
        self.output_dir = Path(node_config.output_dir)
        self.analysis_dir = Path("analysis")

        # Ensure required directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._setup_logging()

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

    def _split_into_sections(self, content: str) -> list[dict]:
        """Split content into logical sections based on headers."""
        lines = content.strip().split('\n')
        sections = []
        current_section = {'header': '', 'content': []}

        for line in lines:
            if line.startswith('#'):
                if current_section['header'] or current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content']).strip()
                    sections.append(current_section)
                    current_section = {'header': '', 'content': []}
                current_section['header'] = line
            else:
                if line.strip():
                    current_section['content'].append(line)

        if current_section['header'] or current_section['content']:
            current_section['content'] = '\n'.join(current_section['content']).strip()
            sections.append(current_section)

        return sections

    def _process_content(self, content: str, base_metadata: dict) -> list[TextNode]:
        """Process content into nodes with structured metadata."""
        try:
            sections = self._split_into_sections(content)
            nodes = []

            for section in sections:
                header = section['header']
                text = section['content']
                full_text = f"{header}\n{text}" if header else text

                # Get hierarchy level from config
                hierarchy_level = self.config.hierarchy_config.get_level_for_text(
                    header if header else full_text
                )

                # Structure the metadata
                metadata = {
                    "pipeline_info": base_metadata["pipeline_info"],
                    "document_info": {
                        "name": base_metadata["document_info"]["name"],
                        "total_pages": base_metadata["document_info"]["total_pages"],
                        "page_num": base_metadata["document_info"]["page_num"]
                    },
                    "content_structure": {
                        "section_header": header,
                        "hierarchy_level": hierarchy_level
                    }
                }

                # Create node
                node = TextNode(
                    text=full_text,
                    metadata=metadata
                )
                nodes.append(node)

            return nodes

        except Exception as e:
            error_msg = f"Error processing content: {str(e)}"
            self.logger.error(error_msg)
            raise TextNodeCreationError(error_msg)

    def create_nodes(self) -> list[TextNode]:
        """Create nodes maintaining document structure."""
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
                    # Structure metadata properly from the start
                    metadata = {
                        "pipeline_info": self.config.base_metadata["pipeline_info"],
                        "document_info": {
                            "name": document_name,
                            "total_pages": len(result["pages"]),
                            "page_num": idx + 1
                        }
                    }

                    nodes = self._process_content(page_dict["md"], metadata)
                    all_nodes.extend(nodes)
                    self.logger.info(f"Created {len(nodes)} nodes for page {idx + 1}")

            if all_nodes:
                self._save_outputs(all_nodes)

            execution_time = time.time() - start_time
            self._log_execution_time(execution_time)

            return all_nodes

        except Exception as e:
            self.logger.error(f"Node creation failed: {str(e)}")
            raise TextNodeCreationError(f"Failed to create nodes: {str(e)}")

    def _save_outputs(self, nodes: list[TextNode]) -> None:
        """Save nodes and basic analysis information."""
        output_path = self.output_dir / f"{self.config.pipeline_name}_nodes.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(nodes, f)

        summary = {
            "total_nodes": len(nodes),
            "pipeline_name": self.config.pipeline_name,
            "node_preview": [
                {
                    "text_preview": node.text[:200] + "...",
                    "metadata": node.metadata,
                    "node_id": node.node_id
                }
                for node in nodes[:5]
            ]
        }

        summary_path = self.output_dir / f"{self.config.pipeline_name}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Saved nodes to {output_path}")
        self.logger.info(f"Saved summary to {summary_path}")

    def _log_execution_time(self, execution_time: float) -> None:
        """Log execution time."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timing_log = f"{timestamp} - Pipeline: {self.config.pipeline_name}, Execution Time: {execution_time:.2f} seconds\n"

        with open(self.analysis_dir / "node_creation_times.txt", 'a') as f:
            f.write(timing_log)