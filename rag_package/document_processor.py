import os
import json
from llama_parse import LlamaParse
from rag_package.errors import DocumentProcessingError
from rag_package.config.parser_config import ParserConfig


class DocumentProcessor:
    """
    Handles document processing operations including parsing PDFs and extracting content.
    This class serves as the first step in the RAG pipeline, converting PDF documentation
    into structured JSON format while preserving document hierarchy and technical details.
    """

    def __init__(self, parsing_config: ParserConfig):
        """
        Initialize the DocumentProcessor with configuration settings.

        The processor uses LlamaParse to convert PDF documents into structured JSON,
        preserving headers, technical specifications, and document hierarchy for
        later processing in the RAG pipeline.

        Args:
            parsing_config: Configuration settings controlling parsing behavior
        """
        self.config = parsing_config
        self.parser = self._initialize_parser()

    def _initialize_parser(self) -> LlamaParse:
        """
        Creates and configures a LlamaParse instance using the current configuration.

        Returns:
            Configured LlamaParse instance ready for document processing
        """
        return LlamaParse(
            result_type=self.config.result_type,
            parsing_instruction=self.config.base_instruction,
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
            if not os.path.exists(self.config.input_data_folder):
                raise DocumentProcessingError(
                    f"Data directory {self.config.input_data_folder} does not exist"
                )

            files = []
            for filename in os.listdir(self.config.input_data_folder):
                filepath = os.path.join(self.config.input_data_folder, filename)
                if os.path.isfile(filepath):
                    files.append(filepath)

            if not files:
                raise DocumentProcessingError(
                    f"No files found in {self.config.input_data_folder}"
                )

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
            self.config.image_dir.mkdir(parents=True, exist_ok=True)
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
            print(f"Saving parsed results to {self.config.results_file}...")
            with open(self.config.results_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print("Results saved successfully!")
        except Exception as e:
            raise DocumentProcessingError(f"Failed to save results: {str(e)}")

    def process_all(self) -> None:
        """
        Execute the complete document processing pipeline.

        Pipeline Steps:
        1. Caching Check:
           - If use_cached_files is True and results exist, skip processing
           - If False or no cached results, proceed with processing

        2. Directory Setup:
           - Ensure image directory exists for storing extracted images

        3. File Collection:
           - Scan input_data_folder for all documents
           - Validate that files exist and directory is not empty

        4. Document Parsing:
           - Initialize LlamaParse with configured settings
           - Process each document while preserving:
             * Document hierarchy (headers, sections)
             * Technical specifications
             * Images and diagrams
             * Tables and structured data
           - Convert documents to structured JSON format

        5. Result Storage:
           - Save processed results to configured results_file location
           - Save any extracted images to image_dir

        6. Summary Generation:
           - Report number of processed documents
           - Confirm save locations for results and images

        The results of this processing stage will be used by the TextNodeCreator
        in the next pipeline stage to create nodes for the vector database.

        Raises:
            DocumentProcessingError: If any processing step fails
        """
        try:
            if self.config.use_cached_files and self.config.results_file.exists():
                print(f"Using existing processed files from: {self.config.results_file}")
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
            print(f"Results saved to: {self.config.results_file}")
            print(f"Images stored in: {self.config.image_dir}")

        except DocumentProcessingError as e:
            print(f"Processing failed: {str(e)}")
            raise