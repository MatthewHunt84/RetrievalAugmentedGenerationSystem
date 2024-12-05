# error.py
class VectorIndexError(Exception):
    """Custom exception for vector index operations"""
    pass

class DocumentProcessingError(Exception):
    """Custom exception for document processing failures"""
    pass

class TextNodeCreationError(Exception):
    """Custom exception for text node creation failures"""
    pass

class QueryEngineError(Exception):
    """Custom exception for query engine failures"""
    pass

class QueryManagerError(Exception):
    """Custom exception for query management failures"""
    pass

