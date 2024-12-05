# error_handler.py
from rag_package.errors import (
    DocumentProcessingError,
    TextNodeCreationError,
    VectorIndexError,
    QueryEngineError,
    QueryManagerError
)

class ErrorHandler:
    """
    Centralizes error handling for the RAG application pipeline. This class manages
    different types of errors, providing appropriate responses and cleanup actions
    for each phase of the application.

    The class uses a mapping between error types and their handlers, making it easy
    to add new error types or modify existing handling strategies without changing
    the main application code.
    """

    def __init__(self):
        # Map each error type to its specific handler method
        self.error_handlers = {
            DocumentProcessingError: self._handle_document_error,
            TextNodeCreationError: self._handle_node_error,
            VectorIndexError: self._handle_index_error,
            QueryEngineError: self._handle_engine_error,
            QueryManagerError: self._handle_query_error
        }

    def handle_error(self, error: Exception, phase: str = None) -> int:
        """
        Main entry point for error handling. Takes an error and optional phase
        information, delegates to the appropriate handler, and returns an exit code.

        Args:
            error: The exception that was raised
            phase: Optional string indicating which phase of the pipeline failed

        Returns:
            int: Exit code (1 for errors, can be customized per error type)
        """
        # Get the specific handler for this error type, or use generic handler
        handler = self.error_handlers.get(type(error), self._handle_generic_error)
        return handler(error, phase)

    def _handle_document_error(self, error: DocumentProcessingError, phase: str) -> int:
        """
        Handles errors during document processing phase. This might include
        cleaning up temporary files or partial processing results.
        """
        print("\nDocument Processing Error")
        print("-------------------------")
        print(f"Failed to process documents: {str(error)}")
        print("Suggested actions:")
        print("- Check if input files exist and are accessible")
        print("- Verify API keys and permissions")
        print("- Ensure sufficient storage space for processing")
        return 1

    def _handle_node_error(self, error: TextNodeCreationError, phase: str) -> int:
        """
        Handles errors during text node creation. This might include
        cleaning up partially created nodes.
        """
        print("\nText Node Creation Error")
        print("----------------------")
        print(f"Failed to create text nodes: {str(error)}")
        print("Suggested actions:")
        print("- Check if parsed results file exists")
        print("- Verify document parsing completed successfully")
        print("- Ensure sufficient memory for node creation")
        return 1

    def _handle_index_error(self, error: VectorIndexError, phase: str) -> int:
        """
        Handles errors during vector index operations. This might include
        cleaning up partial index files.
        """
        print("\nVector Index Error")
        print("-----------------")
        print(f"Failed to create or load vector index: {str(error)}")
        print("Suggested actions:")
        print("- Check if storage directory exists and is writable")
        print("- Verify embedding API credentials")
        print("- Consider recreating the index if corrupted")
        return 1

    def _handle_engine_error(self, error: QueryEngineError, phase: str) -> int:
        """
        Handles errors during query engine setup. This might include
        cleaning up engine-specific resources.
        """
        print("\nQuery Engine Error")
        print("-----------------")
        print(f"Failed to build query engine: {str(error)}")
        print("Suggested actions:")
        print("- Verify all required API keys are set")
        print("- Check model availability and permissions")
        print("- Ensure index is properly initialized")
        return 1

    def _handle_query_error(self, error: QueryManagerError, phase: str) -> int:
        """
        Handles query execution errors. Since queries are typically independent,
        these errors might not require cleanup but should provide clear guidance.
        """
        print("\nQuery Error")
        print("-----------")
        print(f"Query failed: {str(error)}")
        print("Suggested actions:")
        print("- Try rephrasing your question")
        print("- Verify query is not empty")
        print("- Check if referenced content exists in the index")
        return 1

    def _handle_generic_error(self, error: Exception, phase: str) -> int:
        """
        Handles any unexpected errors that don't have specific handlers.
        """
        print("\nUnexpected Error")
        print("----------------")
        print(f"An unexpected error occurred: {str(error)}")
        if phase:
            print(f"Phase: {phase}")
        print("Please report this error to the development team")
        return 1