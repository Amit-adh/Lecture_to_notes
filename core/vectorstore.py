# core/vectorstore.py
from langchain_community.vectorstores import Chroma
from typing import Optional
from app.config import CHROMA_DIR

def get_chroma_vectorstore(embeddings, collection_name: Optional[str] = None, persist: bool = True):
    name = collection_name or "default_collection"
    vs = Chroma(collection_name=name, embedding_function=embeddings, persist_directory=CHROMA_DIR if persist else None)
    return vs

# If you still want FAISS in-memory:
from langchain_community.vectorstores import FAISS
def build_faiss_from_docs(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)
