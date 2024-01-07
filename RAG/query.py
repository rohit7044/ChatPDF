from llama_index.query_engine import RetrieverQueryEngine
from RAG.parent_child_retrieval import PCR_Manager
from llama_index.retrievers import RecursiveRetriever


class QueryEngineManager:
    def __init__(self, index, nodes, service_context):
        self.index = index
        self.nodes = nodes
        self.service_context = service_context

    def context_enhancement(self):
        """
        Enhances the nodes using Parent-Child Chunk Retrieval for better context understanding.

        Returns:
            RecursiveRetriever: A retriever with enhanced context understanding.
        """
        PCR_Processor = PCR_Manager(self.nodes)
        sb_nodes = PCR_Processor.small_to_big_chunk()  # Assumes small_to_big_chunk is defined elsewhere
        all_nodes_dict = {n.node_id: n for n in sb_nodes}
        retriever_chunk = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": self.index.as_retriever(similarity_top_k=2)},
            node_dict=all_nodes_dict,
            verbose=True,
        )
        return retriever_chunk

    def query(self, prompt, pcr=False):
        """
        Executes a query against the configured retriever.

        Args:
            prompt (str): The query prompt.
            pcr (bool, optional): Flag to enable Parent-Child Retrieval. Defaults to False.

        Returns:
            str: The query response.
        """
        # Initialize base retriever with similarity search configuration
        retriever = self.index.as_retriever(similarity_top_k=2)

        # Enhance retriever with context if Parent-Child Retrieval is enabled
        if pcr:
            retriever = self.context_enhancement()

        # Create a query engine with the configured retriever
        query_engine_base = RetrieverQueryEngine.from_args(retriever, service_context=self.service_context)

        # Execute the query and return the response
        response = query_engine_base.query(prompt)
        return str(response)