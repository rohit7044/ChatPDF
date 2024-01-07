from llama_index import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
import chromadb
from RAG.indexing import Indexing
from RAG.model import model


class VectorDatabaseLoad:
    def __init__(self, pdf_file_path):
        self.pdf_file_path = pdf_file_path
        self.db_path = "../upload/chroma.db"  # You can make this an argument if needed
        self.collection_name = "my_collection"  # You can make this an argument if needed

    def load_VectorDB(self):
        """
        Loads the vector database.

        Returns:
            tuple: A tuple containing the vector index metadata database,
                   the service context, and the base nodes.
        """
        # Initialize Database
        db = chromadb.PersistentClient(path=self.db_path)
        chroma_collection = db.get_or_create_collection(name=self.collection_name)

        # Release the embedding model
        Embedding_Manager = model()
        service_context, embed_model = Embedding_Manager.embedding_release()

        # Assign chroma as the vector store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Preprocess the PDF file and get the base nodes
        Doc_Preprocessor = Indexing(self.pdf_file_path)
        base_nodes = Doc_Preprocessor.preprocessing_base()

        # Load the vector index metadata from the vector store
        vector_index_metadata_db = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context, llm=None)

        return vector_index_metadata_db, service_context, base_nodes