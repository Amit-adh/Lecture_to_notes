# core/doc_utils_core.py
import os
import re
import uuid
from pathlib import Path
from typing import Optional, Any, List
from filelock import FileLock

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Pure helper
def clamp_content(content: str, max_doc_chars: int) -> str:
    if len(content) > max_doc_chars:
        return content[:max_doc_chars]
    return content


def get_vectorstore_pure(embeddings: OllamaEmbeddings, session_id: str, chroma_dir: str):
    """
    Return a Chroma vectorstore instance for the given session_id and directory.
    Pure: no caching, no Streamlit.
    """
    vs = Chroma(
        collection_name=f"session_docs_{session_id}",
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )
    return vs


def add_content_to_vectorstore_pure(
    content: str,
    embeddings: OllamaEmbeddings,
    doc_id: str,
    source_name: str,
    vs: Chroma,
    split_fn,
    chroma_dir: str,
):
    """
    Add documents to an existing vectorstore (vs). This function is pure (no st).
    - split_fn(content) -> list[Document]
    - vs is already constructed by caller (e.g., get_vectorstore_pure or cached adapter)
    """
    if not content:
        return

    chunks = split_fn(content)

    for i, d in enumerate(chunks):
        base_meta = dict(d.metadata) if d.metadata else {}
        base_meta.update({"doc_id": doc_id, "source": source_name, "chunk_id": i})
        d.metadata = base_meta

    lock_path = os.path.join(chroma_dir, "chroma.persist.lock")
    lock = FileLock(lock_path)
    with lock:
        vs.add_documents(
            chunks,
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        )
        vs.persist()


def create_rag_chain_pure(retriever: Any, llm: Any, prompt: Optional[ChatPromptTemplate] = None):
    prompt = prompt or ChatPromptTemplate.from_template(DEFAULT_PROMPT)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# small utilities (kept same as before)
def build_stratified_context(full_text: str, max_chars: int) -> str:
    if len(full_text) <= max_chars:
        return full_text

    # sample evenly in many slices (robust to missing late topics)
    parts = 12
    seg_len = max(1, len(full_text) // parts)
    slice_budget = max_chars // parts

    slices = []
    for i in range(parts):
        start = i * seg_len
        end = min(len(full_text), start + slice_budget)
        slices.append(full_text[start:end])

    return "\n\n".join(slices)


def normalize_bullets(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\bSubpoint\s*\d*\s*:\s*', '', text, flags=re.IGNORECASE)
    for ch in ["•", "·", "◦", "▪"]:
        text = text.replace(ch, "\n- ")
    text = text.replace(", - ", "\n- ")
    text = text.replace("; - ", "\n- ")
    lines = text.splitlines()
    fixed_lines = []
    for ln in lines:
        if ln.count("- ") > 1 and not ln.strip().startswith("- "):
            parts = ln.split("- ")
            for p in parts:
                p = p.strip()
                if p:
                    fixed_lines.append("- " + p)
        else:
            fixed_lines.append(ln)
    text = "\n".join(fixed_lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


# Default prompt used by create_rag_chain_pure if prompt not provided
DEFAULT_PROMPT = """
You are a helpful study assistant answering a student's question strictly from the uploaded materials (PDFs, transcripts, or other text).

RULES:
1. Use ONLY the information available in the provided context. If the necessary information is missing, say: 
    "This is not covered in the document."
2. Keep language simple and suitable for a first-year student unless the user explicitly asks for advanced detail.
3. Explain clearly. Use bullet points or short paragraphs when it improves readability.
4. If the question asks for a definition:
- Give a one-line definition.
- Then give 3-6 bullet points of explanation.
5. If the question asks for a comparison (e.g., X vs Y):
- Prefer a short markdown table, then a few bullet points.
6. Do NOT repeat large irrelevant sections from the context.
7. Do NOT hallucinate dates, numbers, or features that are not present in the context.
8. If diagrams/figures are referenced but not visible in text, mention: "Diagram referenced; details not available in the text."

Question:
{question}

Context:
{context}

Answer:
"""
