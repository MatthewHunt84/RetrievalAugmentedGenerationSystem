"""
Configuration module for document hierarchy in the RAG pipeline.

This module defines the HierarchicalConfig class which manages settings
for handling document hierarchy and chunk sizing.
"""
from dataclasses import dataclass, field
import re
from typing import Dict


@dataclass
class HierarchicalConfig:
    """
    Manages document hierarchy through header patterns and chunk sizing.
    Level 1 represents the highest/parent level (h1 headers),
    with deeper levels (2, 3) representing increasingly specific content.

    Attributes:
        header_hierarchy_map: Maps Markdown header levels to hierarchy levels
        chunk_sizes: Configures chunk sizes for different hierarchy levels
        chunk_overlaps: Defines overlap sizes between chunks for each level
    """

    # Map Markdown header levels to hierarchy levels
    header_hierarchy_map: Dict[int, int] = field(default_factory=lambda: {
        1: 1,  # h1 headers (#) → top level 1 (parent)
        2: 2,  # h2 headers (##) → middle level 2
        3: 3,  # h3 headers (###) → lowest level 3
        4: 3,
        5: 3,
        6: 3
    })

    # Configure chunk sizes for different hierarchy levels
    chunk_sizes: Dict[int, int] = field(default_factory=lambda: {
        1: 2048,  # Larger chunks for main sections (parents)
        2: 1024,  # Medium chunks for subsections
        3: 512  # Smaller chunks for detailed content
    })

    # Define overlap to maintain context between chunks
    chunk_overlaps: Dict[int, int] = field(default_factory=lambda: {
        1: 256,  # Larger overlap for main sections
        2: 128,  # Medium overlap for subsections
        3: 64  # Smaller overlap for detailed content
    })

    def get_level_for_text(self, text: str) -> int:
        """
        Determine the hierarchy level of text based on its Markdown header depth.
        Returns 0 for non-header content.

        Args:
            text: The text to analyze for header level

        Returns:
            int: The hierarchy level (0 for non-header content)
        """
        first_line = text.strip().split('\n')[0]
        header_match = re.match(r'^(#+)\s', first_line)

        if not header_match:
            return 0

        header_level = len(header_match.group(1))
        return self.header_hierarchy_map.get(
            min(header_level, max(self.header_hierarchy_map.keys())),
            0
        )