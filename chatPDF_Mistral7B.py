from RAG.storing import VectorDatabaseStore
from RAG.loading import VectorDatabaseLoad
from RAG.query import QueryEngineManager
import os


# Constants
pdf_file_path = ("Source_Documents/A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language "
                 "Models.pdf")
prompt = "give me the summary of hallucination mitigation techniques in Large language Models"
# Only using this because vectorstore requires openAI API key regardless of using local llm
os.environ["OPENAI_API_KEY"] = "Type your API here"


if __name__ == "__main__":
    data_available = False
    pcr = False # Try the parent child chunk retrieval. Not good results to be honest
    if data_available:
        VectorloadManager = VectorDatabaseLoad(pdf_file_path)
        vector_index, service_context, base_nodes = VectorloadManager.load_VectorDB() # Need to remove base_nodes which means no need to re-read the file again.
    else:
        VectorStoreManager = VectorDatabaseStore(pdf_file_path)
        vector_index, service_context, base_nodes = VectorStoreManager.store_vectordb()

    Query_Engine = QueryEngineManager(vector_index, base_nodes, service_context)
    response = Query_Engine.query(prompt)
    print(response)
