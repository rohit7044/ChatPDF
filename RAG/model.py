from llama_index.llms import LlamaCPP
from llama_index.embeddings import resolve_embed_model
from llama_index import ServiceContext
from llama_index.llms import OpenAI

# CONSTANTS
LLM_PATH = "./model/mistral-7b-v0.1.Q4_K_M.gguf"
EMBEDDING_MODEL = "local:BAAI/bge-small-en"
OPENAI_LLM_PATH = "gpt-3.5-turbo"


class model:
    def __init__(self, llm_path=LLM_PATH, embedding_model=EMBEDDING_MODEL):
        self.llm_path = llm_path
        self.embedding_model = embedding_model

    def embedding_release(self):
        """
        Initializes and releases the embedding model and service context.

        Returns:
            tuple: A tuple containing the service context and embedding model.
        """
        # Select Embedding model
        embed_model = resolve_embed_model(self.embedding_model)
        llm = LlamaCPP(model_path=self.llm_path)
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model
        )
        return service_context, embed_model
