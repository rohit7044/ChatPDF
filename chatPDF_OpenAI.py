from RAG.storing import store_VectorDB
from RAG.loading import load_VectorDB
from RAG.query import query_engine
import os

# Constants
pdf_file_path = ("Source_Documents/A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language "
                 "Models.pdf")
prompt = "What are the various hallucination mitigation techniques in Large language Models"

os.environ["OPENAI_API_KEY"] = "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"



if __name__ == "__main__":
    """Please change the model in model.py and make sure to change your API key above"""
    data_availalble = True
    pcr = True # Try the parent child chunk retrieval. Not good results to be honest
    if data_availalble:
        vector_index, service_context, base_nodes = load_VectorDB() # Need to remove base_nodes which means no need to re-read the file again.
    else:
        vector_index, service_context, base_nodes = store_VectorDB(pdf_file_path)

    response = query_engine(vector_index, base_nodes, service_context, prompt)
    print(response)


