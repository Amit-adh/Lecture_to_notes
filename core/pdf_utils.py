# core/pdf_utils.py
from typing import List, Optional
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdf_text(file_path: str) -> Optional[str]:
    """Loads and extracts text from a PDF file quickly (PyMuPDF)."""
    try:
        docs = PyMuPDFLoader(file_path).load()
        pages = []
        for d in docs:
            t = d.page_content.replace("\u00A0", " ").strip()
            if t:
                pages.append(t)
            if not pages:
                return None
        return "\n\n".join(pages)
    except Exception as e:
        return f"__PDF_ERROR__::{e}"

def chunk_text_to_docs(full_text: str, chunk_size=3500, chunk_overlap=400) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.create_documents([full_text])

