# textnode_creator.py
import re
import json
from pathlib import Path
from llama_index.core.schema import TextNode
from rag_package.errors import TextNodeCreationError
from llama_index.core.node_parser import €@#@

class TextNodeCreator:
    """
    Handles the conversion of parsed markdown documents into TextNodes.
    This class manages the process of reading parsed document results,
    matching them with their corresponding images, and creating structured
    TextNodes for further processing.
    """

    def __init__(self, parsed_results_path: str = 'parsed_results.json', image_dir: str = "data_images"):
        """
        Initialize the TextNodeCreator with paths to required resources.

        Args:
            parsed_results_path: Path to the JSON file containing parsed document results
            image_dir: Directory containing the document images
        """
        self.parsed_results_path = Path(parsed_results_path)
        self.image_dir = Path(image_dir)

        # Validate paths immediately to fail fast if they don't exist
        if not self.parsed_results_path.exists():
            raise TextNodeCreationError(
                f"Parsed results file not found at {parsed_results_path}. "
                "Have you run the document parser yet?"
            )
        if not self.image_dir.exists():
            raise TextNodeCreationError(
                f"Image directory not found at {image_dir}"
            )

    def load_parsed_results(self) -> list[dict]:
        """
        Loads previously parsed document results from JSON.

        Returns:
            List of dictionaries containing parsed document data

        Raises:
            TextNodeCreationError: If file reading or JSON parsing fails
        """
        try:
            with open(self.parsed_results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise TextNodeCreationError(
                "The parsed results file appears to be corrupted. "
                "You may need to rerun the document parser."
            )

    def _get_page_number(self, file_name: str) -> int:
        """
        Extracts the page number from an image filename.
        Example: 'manual-page-5.jpg' returns 5

        Args:
            file_name: Name of the image file

        Returns:
            Page number as integer, 0 if no page number found
        """
        match = re.search(r"-page-(\d+)\.jpg$", str(file_name))
        return int(match.group(1)) if match else 0

    def _get_sorted_image_files(self) -> list[Path]:
        """
        Gets all image files from the image directory and sorts them by page number.

        Returns:
            List of Path objects for image files, sorted by page number

        Raises:
            TextNodeCreationError: If image directory is empty or inaccessible
        """
        raw_files = [f for f in self.image_dir.iterdir() if f.is_file()]
        if not raw_files:
            raise TextNodeCreationError(f"No image files found in {self.image_dir}")

        return sorted(raw_files, key=self._get_page_number)

    def create_nodes(self) -> list[TextNode]:
        """
        Creates TextNodes from parsed documents, matching each with its corresponding image.
        Each node contains the text content and metadata about its source and related image.

        Returns:
            List of TextNode objects containing document content and metadata

        Raises:
            TextNodeCreationError: If node creation process fails
        """
        try:
            print("Loading parsed document results...")
            md_json_objs = self.load_parsed_results()

            nodes = []
            print("Creating text nodes from parsed content...")

            for result in md_json_objs:
                # Extract pages and document name
                json_dicts = result["pages"]
                document_name = Path(result["file_path"]).name
                print(f"Processing document: {document_name}")

                # Get text content and corresponding images
                docs = [doc["md"] for doc in json_dicts]
                image_files = self._get_sorted_image_files()

                # Create a node for each page
                for idx, doc in enumerate(docs):
                    try:
                        node = TextNode(
                            text=doc,
                            metadata={
                                "image_path": str(image_files[idx]),
                                "page_num": idx + 1,
                                "document_name": document_name,
                                "total_pages": len(docs)
                            }
                        )
                        nodes.append(node)
                        print(f"Created node for {document_name} - page {idx + 1}/{len(docs)}")
                    except IndexError:
                        print(f"Warning: Missing image for {document_name} page {idx + 1}")
                        continue

            print(f"\nCreated {len(nodes)} text nodes in total")
            return nodes

        except Exception as e:
            raise TextNodeCreationError(f"Failed to create text nodes: {str(e)}")
