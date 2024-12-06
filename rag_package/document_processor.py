import os
import json
from pathlib import Path
from llama_parse import LlamaParse
from rag_package.errors import DocumentProcessingError
from rag_package.rag_config import ParserConfig, input_data_folder


class DocumentProcessor:
    """
    Handles document processing operations including parsing PDFs, managing files,
    and extracting content. This class centralizes all document processing operations
    and maintains consistent file paths and configurations.
    """

    def __init__(self, parsing_config: ParserConfig, data_dir: str = input_data_folder):
        """
        Initialize the DocumentProcessor with a data directory and configuration.

        Args:
            data_dir: Directory containing the files to process
            config: Configuration settings for the parser
        """
        self.data_dir = data_dir
        self.config = parsing_config
        self.parser = self._initialize_parser()

        # Define paths for storing processed data
        self.image_dir = Path("../data_images")
        self.results_file = Path("../parsed_results_old.json")

    def _check_cached_files_exist(self) -> bool:
        """
        Check if both the parsed results and image files exist.
        This is important for determining whether we can use cached results
        or need to reprocess documents.

        Returns:
            bool: True if both parsed results and image directory exist
        """
        return self.results_file.exists() and self.image_dir.exists()

    def _initialize_parser(self) -> LlamaParse:
        """
        Creates and configures a LlamaParse instance using the current configuration.
        This method encapsulates all parser initialization logic in one place.
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
        Retrieves all file paths from the configured data directory.
        This method walks through the data directory and collects paths
        to all files that need processing.

        Returns:
            list[str]: Complete file paths for all files in the directory

        Raises:
            DocumentProcessingError: If the data directory doesn't exist or is empty
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
        Ensures all required directories exist by creating them if necessary.
        This prevents file operation errors later in the processing pipeline.
        """
        try:
            self.image_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise DocumentProcessingError(f"Failed to create required directories: {str(e)}")

    def parse_documents(self, files: list[str]) -> list[dict]:
        """
        Parse PDF documents and return JSON objects.
        This method handles the core document processing using LlamaParse.

        Args:
            files: List of file paths to process

        Returns:
            list[dict]: List of dictionaries containing the parsed document content

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
        This method ensures processed data persists between runs.

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
        This method checks data integrity and ensures saved results are usable.

        Returns:
            list[dict]: The loaded data if verification succeeds

        Raises:
            DocumentProcessingError: If verification fails
        """
        try:
            print("Verifying saved data...")
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Basic validation of loaded data
            if not isinstance(data, list):
                raise DocumentProcessingError("Saved data is not in the expected format")
            if not data:
                raise DocumentProcessingError("Saved data is empty")

            print(f"Successfully verified {len(data)} documents")
            return data
        except json.JSONDecodeError:
            raise DocumentProcessingError("Saved data is corrupted or not valid JSON")
        except Exception as e:
            raise DocumentProcessingError(f"Failed to verify saved data: {str(e)}")

    def extract_images(self, md_json_objs: list[dict]) -> dict:
        """
        Extract images from parsed documents and save them to the image directory.
        This method handles image extraction and storage for use in multimodal processing.

        Args:
            md_json_objs: List of parsed document objects containing image data

        Returns:
            dict: Dictionary containing information about extracted images

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

    def process_all(self, use_cached_files: bool = True) -> None:
        """
        Process all documents in a single call. This is the main processing pipeline
        that coordinates all the individual processing steps.

        Args:
            use_cached_files: If True, will use existing processed files instead of reprocessing
                            when they exist. If False, will always reprocess files.

        Raises:
            DocumentProcessingError: If any processing step fails
        """
        try:
            # Check for existing processed files
            if use_cached_files and self._check_cached_files_exist():
                print("Using existing processed files from:")
                print(f"- Parsed results: {self.results_file}")
                print(f"- Images: {self.image_dir}")

                # Verify the cached data is readable
                try:
                    self.verify_saved_data()
                    return
                except DocumentProcessingError:
                    print("Cached files exist but are corrupted. Proceeding with reprocessing...")

            # If we reach here, we need to process the files
            print("Starting document processing pipeline...")

            # Ensure directories exist
            self.ensure_directories()

            # Get list of files
            files = self.get_data_files()
            if not files:
                raise DocumentProcessingError("No files found in data directory")

            # Parse documents
            print(f"Processing {len(files)} documents...")
            md_json_objs = self.parse_documents(files)

            # Save results
            self.save_results(md_json_objs)

            # Extract images
            print("Extracting images from documents...")
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