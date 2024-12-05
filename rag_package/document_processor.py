# document_processor.py
import os
import json
from pathlib import Path
from dataclasses import dataclass
from llama_parse import LlamaParse
from rag_package.errors import DocumentProcessingError
from . import rag_config

@dataclass
class ParserConfig:
    """Configuration settings for the document parser"""
    result_type: str = rag_config.parser_result_type
    parsing_instruction: str = "You are given IKEA assembly instruction manuals"
    use_vendor_multimodal_model: bool = True
    vendor_multimodal_model_name: str = "anthropic-sonnet-3.5"
    show_progress: bool = True
    verbose: bool = True
    num_workers: int = 8
    language: str = "en"

class DocumentProcessor:
    """
    Handles document processing operations including parsing PDFs, managing files,
    and extracting content. This class centralizes all document processing operations
    and maintains consistent file paths and configurations.
    """

    def __init__(self, data_dir: str = "files", config: ParserConfig = None):
        """
        Initialize the DocumentProcessor with a data directory and configuration.

        Args:
            data_dir: Directory containing the files to process
            config: Configuration settings for the parser. If None, uses defaults
        """
        self.data_dir = data_dir
        self.config = config or ParserConfig()
        self.parser = self._initialize_parser()

        # Define paths for storing processed data
        self.image_dir = Path("../data_images")
        self.results_file = Path("../parsed_results.json")

    def _initialize_parser(self) -> LlamaParse:
        """
        Creates and configures a LlamaParse instance using the current configuration.
        This is a private method as indicated by the underscore prefix.
        """
        return LlamaParse(
            result_type=self.config.result_type,
            parsing_instruction=self.config.parsing_instruction,
            use_vendor_multimodal_model=self.config.use_vendor_multimodal_model,
            vendor_multimodal_model_name=self.config.vendor_multimodal_model_name,
            show_progress=self.config.show_progress,
            verbose=self.config.verbose,
            invalidate_cache=False,
            do_not_cache=False,
            num_workers=self.config.num_workers,
            language=self.config.language
        )

    def get_data_files(self) -> list[str]:
        """
        Retrieves all file paths from the configured data directory.
        Returns a list of complete file paths for all files in the directory.
        """
        files = []
        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            if os.path.isfile(filepath):
                files.append(filepath)
        return files

    def ensure_directories(self) -> None:
        """
        Ensures all required directories exist by creating them if necessary.
        This prevents file operation errors later in the processing pipeline.
        """
        self.image_dir.mkdir(exist_ok=True)

    def parse_documents(self, files: list[str]) -> list[dict]:
        """
        Parse PDF documents and return JSON objects.
        This method makes the expensive API call to process the documents.

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
        Save parsed results to JSON file for future use.

        Args:
            data: List of dictionaries containing parsed document content

        Raises:
            DocumentProcessingError: If saving fails
        """
        try:
            print(f"Saving parsed results to {self.results_file}...")
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            print("Results saved successfully!")
        except Exception as e:
            raise DocumentProcessingError(f"Failed to save results: {str(e)}")

    def verify_saved_data(self) -> list[dict]:
        """
        Verify that saved data can be loaded correctly.
        This is useful for checking data integrity and loading previously processed results.

        Returns:
            The loaded data as a list of dictionaries

        Raises:
            DocumentProcessingError: If verification fails
        """
        try:
            print("Verifying saved data...")
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully verified {len(data)} documents")
            return data
        except Exception as e:
            raise DocumentProcessingError(f"Failed to verify saved data: {str(e)}")

    def extract_images(self, md_json_objs: list[dict]) -> dict:
        """
        Extract images from parsed documents and save them to the image directory.

        Args:
            md_json_objs: List of parsed document objects containing image data

        Returns:
            Dictionary containing information about extracted images

        Raises:
            DocumentProcessingError: If image extraction fails
        """
        try:
            print("Starting image extraction...")
            image_dicts = self.parser.get_images(
                md_json_objs,
                download_path=str(self.image_dir)
            )
            print(f"Successfully extracted images to {self.image_dir}")
            return image_dicts
        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract images: {str(e)}")

    def process_all(self, extract_images: bool = False) -> None:
        """
        Process all documents in a single call. This is the main processing pipeline
        that coordinates all the individual processing steps.

        Args:
            extract_images: Whether to extract images from the documents

        Raises:
            DocumentProcessingError: If any processing step fails
        """
        try:
            # Ensure directories exist
            self.ensure_directories()

            # Get list of files
            files = self.get_data_files()

            # Parse documents
            md_json_objs = self.parse_documents(files)

            # Save results
            self.save_results(md_json_objs)

            # Extract images if requested
            if extract_images:
                self.extract_images(md_json_objs)

            # Verify saved data
            self.verify_saved_data()

            # Print summary
            print("\nProcessing Summary:")
            print(f"Number of documents processed: {len(md_json_objs)}")
            print(f"Results saved to: {self.results_file}")
            print(f"Images stored in: {self.image_dir}")

        except DocumentProcessingError as e:
            print(f"Processing failed: {str(e)}")
            raise

