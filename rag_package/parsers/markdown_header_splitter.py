"""
Markdown header-based document parser for the RAG pipeline.

This module provides a specialized parser that splits documents based on markdown
header levels while maintaining the hierarchical structure of the content.
"""
import re
from typing import Sequence, Tuple
from pydantic import BaseModel, Field
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import TextNode, Document, BaseNode


class MarkdownHeaderSplitter(NodeParser, BaseModel):
    """
    A specialized node parser that splits text based on markdown header levels.

    This parser identifies markdown headers (e.g., #, ##, ###) and uses them to
    create a hierarchical structure of the document content. It inherits from both
    NodeParser for LlamaIndex compatibility and BaseModel for configuration validation.

    Attributes:
        header_level: The header level to split on (1 for h1, 2 for h2, etc.)
        chunk_size: Maximum size of text chunks
        chunk_overlap: Number of characters to overlap between chunks
        include_metadata: Whether to include metadata in nodes
        include_prev_next_rel: Whether to include previous/next relationships
    """

    # Parser configuration
    header_level: int = Field(
        description="The header level to split on (1 for h1, 2 for h2, etc.)"
    )
    chunk_size: int = Field(
        description="Maximum size of text chunks"
    )
    chunk_overlap: int = Field(
        description="Number of characters to overlap between chunks"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in nodes"
    )
    include_prev_next_rel: bool = Field(
        default=True,
        description="Whether to include previous/next relationships"
    )

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True

    def _parse_nodes(
            self,
            nodes: Sequence[BaseNode],
            show_progress: bool = False,
            **kwargs: any
    ) -> list[BaseNode]:
        """
        Required implementation of NodeParser's abstract method.
        Acts as a pass-through since we handle the main processing in get_nodes_from_documents.
        """
        return list(nodes)

    def _split_text_with_headers(self, text: str) -> list[Tuple[int, str, str]]:
        """
        Split text into sections based on markdown headers.

        This method analyzes the text content and breaks it apart at header boundaries,
        maintaining the hierarchical structure of the document. Each section includes
        its header and all content until the next header of the same or higher level.

        Args:
            text: The text content to split

        Returns:
            List of tuples containing:
            - Header level (number of # characters)
            - Header text (without the # marks)
            - Section content (including the header)
        """
        lines = text.split('\n')
        sections = []
        current_section = []
        current_header_level = 0
        current_header = ""

        for line in lines:
            # Look for markdown headers (e.g., "# Header" or "### Subheader")
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())

            if header_match:
                # If we have accumulated content in current_section, save it
                if current_section:
                    sections.append((
                        current_header_level,
                        current_header,
                        '\n'.join(current_section)
                    ))
                    current_section = []

                # Begin a new section with this header
                current_header_level = len(header_match.group(1))
                current_header = header_match.group(2)
                current_section = [line]
            else:
                # Add non-header lines to the current section
                current_section.append(line)

        # Don't forget to add the last section
        if current_section:
            sections.append((
                current_header_level,
                current_header,
                '\n'.join(current_section)
            ))

        return sections

    def _create_node_from_section(
            self,
            text: str,
            metadata: dict,
            header_info: Tuple[int, str]
    ) -> BaseNode:
        """
        Create a node from a section of text with header information.

        This method creates a TextNode instance that includes both the content
        and metadata about its place in the document hierarchy.

        Args:
            text: The section text content
            metadata: Base metadata to include in the node
            header_info: Tuple of (header_level, header_text)

        Returns:
            BaseNode: The created node with enhanced metadata
        """
        header_level, header_text = header_info

        # Combine existing metadata with header information
        enhanced_metadata = {
            **metadata,
            "header_info": {
                "level": header_level,
                "text": header_text
            }
        }

        return TextNode(
            text=text,
            metadata=enhanced_metadata,
            relationships={}  # Will be populated later with parent-child relationships
        )

    def get_nodes_from_documents(
            self,
            documents: list[Document],
            show_progress: bool = False
    ) -> list[BaseNode]:
        """
        Parse documents into nodes based on markdown structure.

        This is the main entry point for document processing. It analyzes each
        document's structure and creates a hierarchical representation using nodes.

        Args:
            documents: List of documents to process
            show_progress: Whether to show progress during processing

        Returns:
            list[BaseNode]: List of created nodes
        """
        nodes = []

        for doc in documents:
            # Split each document into sections based on headers
            sections = self._split_text_with_headers(doc.text)

            # Process sections that match our target header level
            for header_level, header_text, section_text in sections:
                # Check if this section should be processed by this parser level
                if (header_level == self.header_level or
                        (self.header_level == 1 and header_level == 0)):
                    node = self._create_node_from_section(
                        text=section_text,
                        metadata=doc.metadata or {},
                        header_info=(header_level, header_text)
                    )
                    nodes.append(node)

        return nodes