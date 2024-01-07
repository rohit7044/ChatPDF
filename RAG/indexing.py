from pathlib import Path
from llama_hub.file.pdf.base import PDFReader
from llama_index.node_parser import SimpleNodeParser
from llama_index import Document



class Indexing:
    def __init__(self, pdf_file_path):
        self.pdf_file_path = pdf_file_path
        self.base_nodes = []

    def preprocessing_base(self):
        """
        Preprocesses the PDF file by removing arbitrary spaces and parsing the documents into text chunks.

        Returns:
            list: The list of base nodes.
        """
        # Load the file
        loader = PDFReader()
        docs0 = loader.load_data(file=Path(self.pdf_file_path))

        # Preprocessing: Step 1: Remove arbitrary spaces
        doc_text = "\n\n".join([d.get_content() for d in docs0])
        docs = [Document(text=doc_text)]

        # Preprocessing: Step 2: Parsing documents into text chunks also called nodes
        node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)

        base_nodes = node_parser.get_nodes_from_documents(docs)
        for idx, node in enumerate(base_nodes):
            node.id_ = f"node-{idx}"
        self.base_nodes = base_nodes
        return self.base_nodes


