# query_engine.py
from llama_index.core.base.response.schema import Response
from llama_index.core import QueryBundle
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode
from llama_index.core.prompts import PromptTemplate
from typing import List, Optional

# First, let's keep your QA prompt definition
QA_PROMPT_TMPL = """\
You are a chatbot that will help users to get technical responses about an IKEA product manual.

Below we give parsed text from slides in two different formats, as well as the image.

We parse the text in both 'markdown' mode as well as 'raw text' mode. Markdown mode attempts \
to convert relevant diagrams into tables, whereas raw text tries to maintain the rough spatial \
layout of the text.

Use the image information first and foremost. ONLY use the text/markdown information \
if you can't understand the image.

When you reply don't send images links, but only text explanation of that.

Context:
---------------------
{context_str}
---------------------

Given the context information and not prior knowledge, answer the query using ONLY Context information. If you don't find the answer in the Context, reply that you don't know and give a page and document name where the user can find similar information.
Give the page number and the document name where you find the response based on the Context.

Query: {query_str}
Answer: """

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)


class MultimodalQueryEngine(CustomQueryEngine):
    """
    A custom query engine that handles both text and images from IKEA manuals.
    This engine combines document retrieval, image processing, and natural language understanding.
    """

    def __init__(
            self,
            retriever,
            qa_prompt: PromptTemplate,
            multi_modal_llm,
            node_postprocessors: Optional[List] = None
    ):
        super().__init__()
        self._retriever = retriever
        self._qa_prompt = qa_prompt
        self._multi_modal_llm = multi_modal_llm
        self._node_postprocessors = node_postprocessors or []

    @property
    def qa_prompt(self) -> PromptTemplate:
        """Access the QA prompt template."""
        return self._qa_prompt

    @property
    def multi_modal_llm(self):
        """Access the multimodal language model."""
        return self._multi_modal_llm

    @property
    def node_postprocessors(self) -> List:
        """Access the node postprocessors."""
        return self._node_postprocessors

    @property
    def retriever(self):
        """Access the retriever component."""
        return self._retriever

    def _query(self, query_bundle: QueryBundle) -> Response:
        """
        Internal query method that processes the query bundle and returns a response.
        This is called by the parent class's query method.
        """
        return self.custom_query(query_bundle.query_str)

    def custom_query(self, query_str: str) -> Response:
        """
        Process a user's query and return a detailed response.

        This method orchestrates the entire query process:
        1. Retrieves relevant manual sections
        2. Processes any associated images
        3. Applies post-processing if configured
        4. Generates a comprehensive response

        Args:
            query_str: The user's question about the IKEA manual

        Returns:
            Response object containing the answer and source information
        """
        # Step 1: Find relevant manual sections
        nodes = self.retriever.retrieve(query_str)

        # Step 2: Apply any post-processing to improve results
        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(query_str)
            )

        # Step 3: Extract images from the relevant sections
        image_nodes = [
            NodeWithScore(node=ImageNode(image_path=n.node.metadata["image_path"]))
            for n in nodes
        ]

        # Step 4: Combine text from all relevant sections
        ctx_str = "\n\n".join(
            [r.node.get_content(metadata_mode=MetadataMode.LLM).strip() for r in nodes]
        )

        # Step 5: Format the prompt with context and query
        fmt_prompt = self.qa_prompt.format(
            context_str=ctx_str,
            query_str=query_str
        )

        # Step 6: Get response from the multimodal LLM using both text and images
        llm_response = self.multi_modal_llm.complete(
            prompt=fmt_prompt,
            image_documents=[image_node.node for image_node in image_nodes],
        )

        # Step 7: Return formatted response with sources
        return Response(
            response=str(llm_response),
            source_nodes=nodes,
            metadata={"text_nodes": nodes, "image_nodes": image_nodes},
        )