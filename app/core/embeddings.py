# core/embeddings.py
from langchain_community.embeddings import OllamaEmbeddings
from app.config import OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL

def init_embeddings(model_name: str = None):
    model = model_name or OLLAMA_EMBEDDING_MODEL
    return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=model)
