from llama_index.schema import IndexNode
from llama_index.node_parser import SimpleNodeParser


class PCR_Manager:
    def __init__(self, base_nodes):
        """
        Initialize the NodeChunkManager with the base nodes.

        Args:
            base_nodes (list): A list of base nodes to chunk into smaller nodes.
        """
        if not base_nodes:
            raise ValueError("Base nodes cannot be empty.")
        self.base_nodes = base_nodes

    def small_to_big_chunk(self):
        """
        Generates smaller chunks of nodes and creates parent-child relationships.

        Returns:
            list: List of all nodes, including the original nodes and smaller sub-nodes.

        Raises:
            ValueError: If no base nodes are found. Initialize class with base nodes first.
        """
        # Define the chunk sizes and parse it to nodes
        sub_chunk_sizes = [256, 512]
        sub_node_parsers = [SimpleNodeParser.from_defaults(chunk_size=c) for c in sub_chunk_sizes]

        all_nodes = []
        # Access each node and create parent child relationship
        for base_node in self.base_nodes:
            for n in sub_node_parsers:
                sub_nodes = n.get_nodes_from_documents([base_node])
                sub_inodes = [IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes]
                all_nodes.extend(sub_inodes)

            # also add original node to node
            original_node = IndexNode.from_text_node(base_node, base_node.node_id)
            all_nodes.append(original_node)

        return all_nodes