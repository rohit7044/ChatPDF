from llama_index import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
import chromadb
from RAG.indexing import Indexing
from RAG.model import model


class VectorDatabaseStore:
    def __init__(self, pdf_file_path):
        self.pdf_file_path = pdf_file_path
        self.db_path = "./upload/chroma.db"  # or parameterize this if needed
        self.collection_name = "my_collection"  # or parameterize this if needed
        self.vector_index_metadata_db = None
        self.service_context = None
        self.base_nodes = None

    def store_vectordb(self):
        """
        Stores the vector database.

        Returns:
            tuple: A tuple containing the vector index metadata database,
                   the service context, and the base nodes.
        """
        # Initialize Database
        db = chromadb.PersistentClient(path=self.db_path)
        chroma_collection = db.get_or_create_collection(name=self.collection_name)

        # Release the embedding model
        Embedding_Manager = model()
        self.service_context, embed_model = Embedding_Manager.embedding_release()

        # Assign chroma as the vector store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Preprocess the PDF file and get the base nodes
        Doc_Preprocessor = Indexing(self.pdf_file_path)
        self.base_nodes = Doc_Preprocessor.preprocessing_base()

        # Create index (assuming this also stores the data)
        self.vector_index_metadata_db = VectorStoreIndex(
            self.base_nodes,
            storage_context=storage_context,
            service_context=self.service_context)

        return self.vector_index_metadata_db, self.service_context, self.base_nodes

