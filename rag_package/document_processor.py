import os
import json
from pathlib import Path
from llama_parse import LlamaParse
from rag_package.errors import DocumentProcessingError
from rag_package.rag_config import ParserConfig, input_data_folder


class DocumentProcessor:
    """
    Handles document processing operations including parsing PDFs and extracting content.
    This class serves as the first step in the RAG pipeline, converting PDF documentation
    into structured JSON format while preserving document hierarchy and technical details.
    """

    def __init__(self, parsing_config: ParserConfig, data_dir: str = input_data_folder):
        """
        Initialize the DocumentProcessor with configuration settings and data directory.

        The processor uses LlamaParse to convert PDF documents into structured JSON,
        preserving headers, technical specifications, and document hierarchy for
        later processing in the RAG pipeline.

        Args:
            parsing_config: Configuration settings controlling parsing behavior
            data_dir: Directory containing the PDF files to process
        """
        self.data_dir = data_dir
        self.config = parsing_config
        self.parser = self._initialize_parser()

        # Define paths for storing processed data
        self.image_dir = Path("data_images")
        self.results_file = Path("parsed_results.json")

    def _initialize_parser(self) -> LlamaParse:
        """
        Creates and configures a LlamaParse instance using the current configuration.

        The parser is configured to preserve document structure and technical content,
        preparing the data for subsequent node creation in the RAG pipeline.
        """
        return LlamaParse(
            result_type=self.config.result_type,
            parsing_instruction=self.config.parsing_instruction,
            use_vendor_multimodal_model=self.config.use_vendor_multimodal_model,
            vendor_multimodal_model_name=self.config.vendor_multimodal_model_name,
            show_progress=self.config.show_progress,
            verbose=self.config.verbose,
            invalidate_cache=self.config.invalidate_cache,
            do_not_cache=self.config.do_not_cache,
            num_workers=self.config.num_workers,
            language=self.config.language
        )

    def get_data_files(self) -> list[str]:
        """
        Retrieves file paths from the configured data directory.

        Returns:
            List of complete file paths for all files in the directory.

        Raises:
            DocumentProcessingError: If the data directory doesn't exist or is empty.
        """
        try:
            if not os.path.exists(self.data_dir):
                raise DocumentProcessingError(f"Data directory {self.data_dir} does not exist")

            files = []
            for filename in os.listdir(self.data_dir):
                filepath = os.path.join(self.data_dir, filename)
                if os.path.isfile(filepath):
                    files.append(filepath)

            if not files:
                raise DocumentProcessingError(f"No files found in {self.data_dir}")

            return files
        except Exception as e:
            raise DocumentProcessingError(f"Error accessing data files: {str(e)}")

    def ensure_directories(self) -> None:
        """
        Creates required directories if they don't exist.

        Raises:
            DocumentProcessingError: If directory creation fails
        """
        try:
            self.image_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise DocumentProcessingError(f"Failed to create required directories: {str(e)}")

    def parse_documents(self, files: list[str]) -> list[dict]:
        """
        Parse PDF documents into structured JSON format.

        The parsing process preserves document structure, headers, and technical content,
        preparing the data for subsequent node creation in the RAG pipeline.

        Args:
            files: List of file paths to process

        Returns:
            List of dictionaries containing the parsed document content

        Raises:
            DocumentProcessingError: If parsing fails
        """
        try:
            print(f"Starting PDF content extraction for {len(files)} files...")
            md_json_objs = self.parser.get_json_result(files)
            print(f"Successfully processed {len(md_json_objs)} documents")
            return md_json_objs
        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse documents: {str(e)}")

    def save_results(self, data: list[dict]) -> None:
        """
        Save parsed results to JSON file for use in subsequent pipeline stages.

        Args:
            data: List of dictionaries containing parsed document content

        Raises:
            DocumentProcessingError: If saving fails
        """
        try:
            print(f"Saving parsed results to {self.results_file}...")
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print("Results saved successfully!")
        except Exception as e:
            raise DocumentProcessingError(f"Failed to save results: {str(e)}")

    def process_all(self, use_cached_files: bool = True) -> None:
        """
        Execute the complete document processing pipeline.

        This method orchestrates the entire parsing process, from reading input files
        to saving the structured output. It includes caching logic to avoid
        reprocessing previously parsed documents.

        Args:
            use_cached_files: Whether to use existing processed files if available

        Raises:
            DocumentProcessingError: If any processing step fails
        """
        try:
            if use_cached_files and self.results_file.exists():
                print(f"Using existing processed files from: {self.results_file}")
                return

            print("Starting document processing pipeline...")
            self.ensure_directories()

            files = self.get_data_files()
            if not files:
                raise DocumentProcessingError("No files found in data directory")

            print(f"Processing {len(files)} documents...")
            md_json_objs = self.parse_documents(files)

            self.save_results(md_json_objs)

            print("\nProcessing Summary:")
            print(f"Documents processed: {len(md_json_objs)}")
            print(f"Results saved to: {self.results_file}")
            print(f"Images stored in: {self.image_dir}")

        except DocumentProcessingError as e:
            print(f"Processing failed: {str(e)}")
            raise