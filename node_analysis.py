import pickle
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import logging


class NodeAnalyzer:
    """
    A class to analyze and compare RAG pipeline node outputs stored in pickle files.

    This class examines the structure and content of document nodes, making it
    easier to compare different ingestion approaches and understand how documents
    are being processed. Analysis results are stored in an 'analysis' directory
    at the root level of the project.
    """

    def __init__(self, output_dir: str = "analysis"):
        """
        Initialize the analyzer with an output directory for saving results.

        Args:
            output_dir: Directory where analysis results will be saved, defaults to 'analysis'
        """
        # Create Path object for the project root (assuming this file is in the root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Add a file handler for the log
        log_file = self.output_dir / "analysis_log.txt"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def load_pickle(self, file_path: str) -> any:
        """Load a pickle file and return its contents."""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading pickle file {file_path}: {str(e)}")
            raise

    def analyze_hierarchy_levels(self, pickle_files: str | list[str]) -> pd.DataFrame:
        """
        Analyze hierarchy levels found in the document(s).

        Args:
            pickle_files: Single pickle file path or list of paths to analyze

        Returns:
            DataFrame containing hierarchy level statistics for each pipeline
        """
        if isinstance(pickle_files, str):
            pickle_files = [pickle_files]

        results = []

        for file_path in pickle_files:
            pipeline_name = Path(file_path).stem
            index_data = self.load_pickle(file_path)

            # Get the nodes list, handling different data structures
            nodes = (index_data.index_struct.nodes
                     if hasattr(index_data, 'index_struct')
                     else index_data)

            # Extract hierarchy levels from all nodes
            hierarchy_levels = []
            for node in nodes:
                try:
                    # Handle both dictionary-like and object-like nodes
                    if isinstance(node, dict):
                        hierarchy_level = node.get('metadata', {}).get('hierarchy_level')
                    else:
                        hierarchy_level = node.metadata.get('hierarchy_level')
                    hierarchy_levels.append(hierarchy_level)
                except AttributeError:
                    continue

            # Calculate statistics
            level_stats = {
                'pipeline_name': pipeline_name,
                'total_nodes': len(nodes),
                'unique_hierarchy_levels': len(set(filter(None, hierarchy_levels))),
                'max_hierarchy_level': max(filter(None, hierarchy_levels), default=0),
                'level_distribution': pd.Series(hierarchy_levels).value_counts().to_dict()
            }

            results.append(level_stats)

            # Log the analysis
            self.logger.info(f"Analyzed hierarchy levels for {pipeline_name}")

        # Create DataFrame from results
        df = pd.DataFrame(results)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"hierarchy_analysis_{timestamp}.csv"
        df.to_csv(output_file, index=False)

        return df

    def get_page_nodes(self, pickle_files: str | list[str], page_number: int) -> dict:
        """
        Retrieve all nodes associated with a specific page number.

        Args:
            pickle_files: Single pickle file path or list of paths to analyze
            page_number: Page number to search for

        Returns:
            Dictionary containing nodes and their metadata for the specified page
        """
        if isinstance(pickle_files, str):
            pickle_files = [pickle_files]

        page_results = {}

        for file_path in pickle_files:
            pipeline_name = Path(file_path).stem
            index_data = self.load_pickle(file_path)

            # Get the nodes list, handling different data structures
            nodes = (index_data.index_struct.nodes
                     if hasattr(index_data, 'index_struct')
                     else index_data)

            # Find nodes for the specified page
            page_nodes = []
            for node in nodes:
                try:
                    # Handle both dictionary-like and object-like nodes
                    if isinstance(node, dict):
                        metadata = node.get('metadata', {})
                        text = node.get('text', '')
                        node_id = node.get('id', '')
                    else:
                        metadata = node.metadata
                        text = node.text
                        node_id = node.node_id

                    if metadata.get('page_num') == page_number:
                        node_info = {
                            'text_preview': text[:200] + '...' if len(text) > 200 else text,
                            'metadata': metadata,
                            'node_id': node_id
                        }
                        page_nodes.append(node_info)
                except AttributeError:
                    continue

            page_results[pipeline_name] = {
                'total_nodes_for_page': len(page_nodes),
                'nodes': page_nodes
            }

            # Log the analysis
            self.logger.info(f"Found {len(page_nodes)} nodes for page {page_number} in {pipeline_name}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"page_{page_number}_analysis_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(page_results, f, indent=2)

        return page_results


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = NodeAnalyzer()

    # Example paths - replace with your actual paths
    pickle_files = ["node_outputs/first_pipeline_nodes.pkl"]

    # Analyze hierarchy levels
    hierarchy_results = analyzer.analyze_hierarchy_levels(pickle_files)
    print("\nHierarchy Level Analysis:")
    print(hierarchy_results)

    # Get nodes for a specific page
    page_results = analyzer.get_page_nodes(pickle_files, page_number=10)
    print("\nPage 10 Nodes:")
    print(json.dumps(page_results, indent=2))