#!/usr/bin/env python3
"""
Importable module with functions to:
 - load PDF text
 - chunk text
 - build a vectorstore (FAISS)
 - create a RAG chain
 - run an interactive ask loop (optional helper)

"""

import os
from typing import List, Optional

from dotenv import load_dotenv

# Document loading / splitting
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vectorstore / embeddings / LLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

# Prompt composition
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -----------------------
# Core functions (importable)
# -----------------------
def load_pdf_text(path: str) -> Optional[str]:
    """Load PDF pages into a single string. Returns None on failure or if no text found."""
    try:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        pages = []
        for d in docs:
            t = d.page_content.replace("\u00A0", " ").strip()
            if t:
                pages.append(t)
        if not pages:
            return None
        return "\n\n".join(pages)
    except Exception as e:
        # For an importable module, raise or return None — choose None for simple integration
        print(f"[pdf_summarizer] Error loading PDF: {e}")
        return None


def chunk_documents_from_text(full_text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Return LangChain Document objects split from full_text."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.create_documents([full_text])


def init_ollama_embeddings(model_name: Optional[str] = None) -> OllamaEmbeddings:
    model = model_name or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    return OllamaEmbeddings(model=model)


def build_faiss_vectorstore(docs, embeddings: OllamaEmbeddings) -> FAISS:
    """Create an in-memory FAISS vectorstore from docs using the provided embeddings."""
    return FAISS.from_documents(docs, embeddings)


def make_prompt_template(template: Optional[str] = None) -> ChatPromptTemplate:
    t = template or """
    You are a helpful study assistant that answers questions using ONLY the provided CONTEXT.
Follow these rules exactly:

1) SOURCE RULE
- Use information only from the Context. If the answer is not present in the Context, reply exactly: "This is not covered in the document."

2) AUDIENCE & STYLE
- Write for a first-year undergraduate: simple language, short sentences.
- Prefer bullet lists or short paragraphs for clarity.

3) STRUCTURE RULES
- If the user asks for a definition:
  - Give a one-line definition.
  - Then give 3–5 short bullet points that clarify or give an example.
- If the user asks for a comparison:
  - Prefer a short markdown table (2–5 rows), then 2–4 bullet points summarizing key differences.
- Do NOT repeat long verbatim blocks from the Context; summarize instead.

4) SAFETY & ACCURACY
- Do NOT hallucinate facts, numbers, or dates.
- If a diagram is referenced but not present, say: "Diagram referenced; details not available in the text."

5) LENGTH
- Keep answers concise. If the user asks for more depth, the model may expand on request.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""
    return ChatPromptTemplate.from_template(t)


def create_rag_chain(retriever, llm: ChatOllama, prompt_template: Optional[str] = None):
    """Compose and return a RAG runnable chain (invoke-able via .invoke)."""
    prompt = make_prompt_template(prompt_template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def init_llm(model_name: Optional[str] = None, base_url: Optional[str] = None, **kwargs) -> ChatOllama:
    """
    Initialize ChatOllama (or swap this factory to create another LLM wrapper).
    Pass kwargs like temperature, num_ctx, num_predict if supported.
    """
    m = model_name or os.getenv("OLLAMA_LLM_MODEL", "koesn/llama3-8b-instruct:latest")
    # ChatOllama in your env may accept different params; adapt as needed
    return ChatOllama(model=m)


# -----------------------
# Optional convenience helper (interactive loop)
# -----------------------
def ask_loop(rag_chain):
    """Simple CLI ask loop useful for local debugging. Importable but interactive."""
    if rag_chain is None:
        print("RAG chain not provided.")
        return
    print("\n✅ Ready. Ask any question about the PDF (type 'exit' to quit):")
    while True:
        q = input("❓ Question: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        try:
            print("🧠 Generating answer...\n")
            out = rag_chain.invoke(q)
            print(out)
        except Exception as e:
            print(f"[pdf_summarizer] Error during generation: {e}")
        print("-" * 50)


# -----------------------
# Example pipeline helper you can call from Streamlit or tests
# -----------------------
def build_rag_from_pdf(pdf_path: str, embedding_model: Optional[str] = None, llm_model: Optional[str] = None):
    """
    High-level helper: load PDF -> chunk -> embed -> build retriever -> return rag_chain and vectorstore
    Returns (rag_chain, vectorstore, llm, embeddings)
    """
    load_dotenv()

    text = load_pdf_text(pdf_path)
    if not text:
        raise RuntimeError("No text loaded from PDF (scanned/empty or loading error).")

    docs = chunk_documents_from_text(text)
    embeddings = init_ollama_embeddings(embedding_model)
    vs = build_faiss_vectorstore(docs, embeddings)
    retriever = vs.as_retriever()

    llm = init_llm(llm_model)
    rag_chain = create_rag_chain(retriever, llm)
    return rag_chain, vs, llm, embeddings
