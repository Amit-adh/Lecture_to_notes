# main.py
import os
import uuid
from pathlib import Path

# --- for max file sizes ---
MAX_FILE_MB = 100

# ---- CPU/BLAS threads (set BEFORE heavy imports) ----
MAX_THREADS = int(os.environ.get("MAX_THREADS", "4"))

os.environ.setdefault("OMP_NUM_THREADS", str(MAX_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(MAX_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(MAX_THREADS))

import streamlit as st
import io
import torch
import subprocess
import hashlib
from typing import Optional, List

# --- ASR (fast) ---
from faster_whisper import WhisperModel

# --- LLM / RAG ---
from transformers import logging as hf_logging  # quiet HF logs (not used for ASR now)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
# Switched to PyMuPDFLoader for speed/robustness
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Global Configuration
# -----------------------------
st.set_page_config(page_title="AI Document & Media Q&A (Fast)", layout="wide")

UPLOAD_DIRECTORY = os.environ.get("UPLOAD_DIR", "uploads")
TRANSCRIPT_CACHE_DIR = os.environ.get("TRANSCRIPT_CACHE_DIR", "transcripts")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2:1b-instruct-q4_K_M")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2:1b")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "2048"))  # >>> now actually used

# ASR model sizing (adjust for your machine)
ASR_MODEL_SIZE = os.environ.get("ASR_MODEL_SIZE", "small")  # "distil-medium.en" | "base" | "small" | "medium"
ASR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_COMPUTE = os.environ.get("ASR_COMPUTE", "float16" if ASR_DEVICE == "cuda" else "int8")

# Ensure dirs
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(TRANSCRIPT_CACHE_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Speed up PyTorch ops on CPU
try:
    torch.set_num_threads(MAX_THREADS)
except Exception:
    pass

# Quiet down HF warnings
hf_logging.set_verbosity_error()


# -----------------------------
# Helpers
# -----------------------------

def cleanup_upload_dir():
    if not os.path.exists(UPLOAD_DIRECTORY):
        return
    for p in Path(UPLOAD_DIRECTORY).glob("*"):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass


def save_uploaded_file(uploaded_file) -> str:
    if uploaded_file.size > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"File too large (>{MAX_FILE_MB} MB).")

    ext = Path(uploaded_file.name).suffix.lower()
    allowed_exts = {".mp3", ".wav", ".mp4", ".mov", ".pdf"}
    if ext not in allowed_exts:
        raise ValueError("Unsupported file type.")

    uid = uuid.uuid4().hex
    safe_name = f"{uid}{ext}"
    file_path = os.path.join(UPLOAD_DIRECTORY, safe_name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path, safe_name


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def cached_transcript_path(content_hash: str) -> str:
    return os.path.join(TRANSCRIPT_CACHE_DIR, f"{content_hash}.txt")


def normalized_wav_cache_path(content_hash: str) -> str:
    return os.path.join(UPLOAD_DIRECTORY, f"{content_hash}_16k.wav")


def ffmpeg_to_wav16k_mono(src: str, dst: str):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src, "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", "16000", "-f", "wav", dst
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300)  # 5 min cap
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio conversion timed out.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e}")


def normalize_to_wav16k(input_path: str, content_hash: str) -> str:
    # Cache normalized WAV once per file hash
    wav_path = normalized_wav_cache_path(content_hash)
    if os.path.exists(wav_path):
        return wav_path
    ffmpeg_to_wav16k_mono(input_path, wav_path)
    return wav_path


# -----------------------------
# Cached Resources
# -----------------------------
@st.cache_resource
def load_asr_model():
    st.write("⏳ Cache miss: loading faster-whisper...")
    model = WhisperModel(
        ASR_MODEL_SIZE,
        device=ASR_DEVICE,
        compute_type=ASR_COMPUTE,
    )
    st.write(f"✅ faster-whisper loaded ({ASR_MODEL_SIZE}, {ASR_DEVICE}/{ASR_COMPUTE})")
    return model


@st.cache_resource
def load_ollama_llm():
    st.write(f"⏳ Cache miss: initializing Ollama LLM '{OLLAMA_LLM_MODEL}'...")
    try:
        # >>> use global OLLAMA_NUM_PREDICT as the default limit
        num_predict = OLLAMA_NUM_PREDICT

        llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_LLM_MODEL,
            temperature=0.2,
            num_ctx=4096, #previously 2048
            num_predict=num_predict,
        )
        # Warmup (very small)
        _ = llm.invoke("hello")[:1]
        st.write(f"✅ Ollama LLM ready. (num_predict={num_predict})")
        return llm
    except Exception as e:
        st.error(f"❌ Ollama LLM init failed. Is Ollama running? Error: {e}")
        return None


@st.cache_resource
def load_ollama_embeddings():
    st.write(f"⏳ Cache miss: initializing Ollama embeddings '{OLLAMA_EMBEDDING_MODEL}'...")
    try:
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_EMBEDDING_MODEL,
        )
        st.write("✅ Ollama embeddings ready.")
        return embeddings
    except Exception as e:
        st.error(f"❌ Ollama embeddings init failed. Is Ollama running? Error: {e}")
        return None


# -----------------------------
# Cached Data Helpers
# -----------------------------
@st.cache_data
def split_text_cached(content: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=400,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.create_documents([content])


@st.cache_data
def load_pdf_text_cached(file_path: str) -> Optional[str]:
    """Loads and extracts text from a PDF file quickly (PyMuPDF)."""
    try:
        docs = PyMuPDFLoader(file_path).load()
        pages = []
        for d in docs:
            t = d.page_content.replace("\u00A0", " ").strip()
            if t:
                pages.append(t)
        return "\n\n".join(pages)
    except Exception as e:
        return f"__PDF_ERROR__::{e}"


# -----------------------------
# Core Processing
# -----------------------------
def transcribe_media_file_fast(asr_model: WhisperModel, file_path: str, cache_key: str, lang: Optional[str]) -> Optional[str]:
    """
    Convert media → 16k mono WAV and transcribe with faster-whisper.
    Uses VAD to skip silence. Caches transcript by file hash.
    """
    cache_path = cached_transcript_path(cache_key)
    if os.path.exists(cache_path):
        st.info("☑️ Using cached transcript.")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    st.info("🎙️ Starting transcription...")

    try:
        wav_path = normalize_to_wav16k(file_path, cache_key)
    except Exception as e:
        st.error(f"Audio preprocessing failed: {e}")
        return None

    # If lang is None → autodetect; else force (speeds up if known)
    transcribe_kwargs = dict(
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=1,
        best_of=1,
        temperature=0,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        task="transcribe",
    )
    if lang and lang.lower() != "auto":
        transcribe_kwargs["language"] = lang

    with st.spinner("Transcribing (faster-whisper + VAD)..."):
        try:
            segments, info = asr_model.transcribe(wav_path, **transcribe_kwargs)
        except Exception as e:
            st.error(f"Transcription error: {e}")
            return None

        texts: List[str] = [seg.text for seg in segments]
        final_text = " ".join(texts).strip()

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    st.success("Transcription complete.")
    return final_text


# ---- Document / RAG helpers ----

MAX_DOC_CHARS = int(os.environ.get("MAX_DOC_CHARS", "200000"))
INITIAL_SUMMARY_MAX_CHARS = int(os.environ.get("INITIAL_SUMMARY_MAX_CHARS", "16000"))


def clamp_content(content: str) -> str:
    """
    Limit document length to avoid huge memory / embedding time.
    """
    if len(content) > MAX_DOC_CHARS:
        st.warning(f"Document too large, truncating to first {MAX_DOC_CHARS} characters for speed.")
        return content[:MAX_DOC_CHARS]
    return content


session_id = st.session_state.get("session_id")
if not session_id:
    session_id = uuid.uuid4().hex
    st.session_state.session_id = session_id


@st.cache_resource
def get_vectorstore(embeddings):
    """
    Shared Chroma vectorstore for the whole session / app.
    All PDFs (and optionally transcripts) go into this single collection.
    """
    vs = Chroma(
        collection_name=f"session_docs_{session_id}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vs


def add_content_to_vectorstore(
    content: str,
    embeddings: OllamaEmbeddings,
    doc_id: str,
    source_name: str,
):
    """
    Add a single document's content into the shared vectorstore.
    - content: raw text (already loaded from PDF or transcript)
    - doc_id: stable id for this document (e.g. SHA-256 hash)
    - source_name: original filename (for metadata/debug)
    """
    if not content:
        return

    content = clamp_content(content)
    chunks = split_text_cached(content)  # list[Document]

    # Attach metadata directly to each chunk Document
    for i, d in enumerate(chunks):
        base_meta = dict(d.metadata) if d.metadata else {}
        base_meta.update({"doc_id": doc_id, "source": source_name, "chunk_id": i})
        d.metadata = base_meta

    vs = get_vectorstore(embeddings)

    # Do NOT pass metadatas kwarg to avoid the multiple-values error
    vs.add_documents(
        chunks,
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
    )
    vs.persist()


def create_rag_chain(llm: Ollama, embeddings: OllamaEmbeddings):
    """
    Create a RAG chain over *all* documents currently stored
    in the shared vectorstore.
    Call this after you've added at least one document via add_content_to_vectorstore.
    """
    vs = get_vectorstore(embeddings)

    with st.spinner("Building retriever and RAG chain from the uploaded materials..."):
        retriever = vs.as_retriever(
            search_type="mmr",
            # pull more chunks so answers can see more of the document
            search_kwargs={"k": 20, "fetch_k": 120, "lambda_mult": 0.25},
        )

        prompt_template = """
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

        prompt = ChatPromptTemplate.from_template(prompt_template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    return rag_chain



def generate_initial_notes_if_needed():
    """
    If the set of documents in doc_ids has changed since the last time
    we generated notes, generate a compact initial overview:
      1) main topics (bullet list)
      2) short study summary (bullet list)
    using the RAW document text (not the retriever),
    and append it as an assistant message.
    """
    if not llm:
        return
    if not st.session_state.doc_ids:
        return

    current_state = frozenset(st.session_state.doc_ids)
    prev_state = st.session_state.get("last_notes_doc_state")

    # Only generate notes if the underlying document set has changed
    if current_state == prev_state:
        return

    # Build combined raw text from all docs we currently know about
    combined_parts = []
    for doc_id in st.session_state.doc_ids:
        txt = st.session_state.raw_docs.get(doc_id)
        if txt:
            combined_parts.append(txt)

    if not combined_parts:
        return

    combined_text = "\n\n".join(combined_parts)
    if len(combined_text) > INITIAL_SUMMARY_MAX_CHARS:
        combined_text = combined_text[:INITIAL_SUMMARY_MAX_CHARS]

    auto_prompt = f"""
You are extracting a high-level overview from the following course materials.

[Context Start]
{combined_text}
[Context End]

Do TWO things in this exact order:

1) Main Topics
- Output ONLY a bullet list of 6–12 main topics.
- Each bullet must be a short topic title (2–6 words).
- NO explanations.
- NO sub-bullets.

2) Short Study Summary
- Then write 6–10 bullet points.
- Each bullet must be exactly one clear sentence.
- Cover the ENTIRE material from start to end (not just one section).
- Do NOT repeat the same idea in multiple bullets.
- Keep it beginner-friendly.

Do NOT add any extra sections or headings. Just:
- Bullet list of main topics
- Then bullet list of summary points
"""

    try:
        with st.spinner("Generating initial topics and short summary from the uploaded materials..."):
            notes = llm.invoke(auto_prompt)
    except Exception as e:
        st.error(f"Error while generating initial notes: {e}")
        return

    # Store the notes as an assistant message so they appear before user questions
    st.session_state.messages.append({
        "role": "assistant",
        "content": notes,
    })

    # Remember that we've generated notes for this exact document set
    st.session_state.last_notes_doc_state = current_state


def stream_answer(chain, question: str):
    """
    Stream tokens from the final LLM stage through the composed chain.
    """
    try:
        for chunk in chain.stream(question):
            # chunk is already parsed into string via StrOutputParser in the chain
            yield str(chunk)
    except Exception as e:
        # Surface a readable error instead of crashing the whole app
        yield f"\n[Error during generation: {e}]"


# Run this once per server process
cleanup_upload_dir()

# -----------------------------
# UI State Init
# -----------------------------
# -----------------------------
# UI State Init
# -----------------------------
if "doc_ids" not in st.session_state:
    st.session_state.doc_ids = set()  # set is easier than []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "content_hash" not in st.session_state:
    st.session_state.content_hash = None
if "last_notes_doc_state" not in st.session_state:
    st.session_state.last_notes_doc_state = None  # to track when notes were last generated
if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = {}  # doc_id -> full raw text



# -----------------------------
# Sidebar: Model Status
# -----------------------------
with st.sidebar:
    st.header("Models & Settings")
    with st.status("Initializing...", expanded=False) as status:
        asr_model = load_asr_model()
        llm = load_ollama_llm()
        embeddings = load_ollama_embeddings()

        if all([asr_model, llm, embeddings]):
            status.update(label="✅ Models Ready", state="complete", expanded=False)
        else:
            status.update(label="⚠️ Model init failed", state="error", expanded=True)
            st.error("Please ensure Ollama is running and models exist (ollama pull ...).")

    st.subheader("ASR Language")
    asr_language = st.selectbox(
        "Pick language (Auto is fine if unsure):",
        options=["Auto", "en", "hi", "es", "fr", "de", "zh", "ja"],
        index=0
    )
    asr_language = None if asr_language == "Auto" else asr_language

    st.subheader("Ollama Options")
    st.caption("Configured in code; threads capped to CPU cores for speed.")
    st.write(
        f"- Base URL: {OLLAMA_BASE_URL}\n"
        f"- LLM: `{OLLAMA_LLM_MODEL}`\n"
        f"- Embeddings: `{OLLAMA_EMBEDDING_MODEL}`\n"
        f"- num_predict: {OLLAMA_NUM_PREDICT}"
    )
    st.subheader("ASR Settings")
    st.write(f"- Model: `{ASR_MODEL_SIZE}` · Device: `{ASR_DEVICE}` · Compute: `{ASR_COMPUTE}`")


# -----------------------------
# Abort if not all are loaded.
# -----------------------------
if not all([asr_model, llm, embeddings]):
    st.error("Models not ready. Please fix server configuration.")
    st.stop()

# -----------------------------
# Main UI
# -----------------------------
st.title("Lectures and Slides to Notes")
st.markdown("Upload a media file (`mp3`, `wav`, `mp4`, `mov`) or one/more documents (`pdf`) and ask questions about their content.")

st.header("1) Choose File Type")
file_type = st.radio("Select file type:", ("Media (Audio/Video)", "Document (PDF)"), horizontal=True)

accepted_types = ["mp3", "wav", "mp4", "mov"] if file_type.startswith("Media") else ["pdf"]

media_file = None
pdf_files = None

if file_type.startswith("Media"):
    media_file = st.file_uploader("Upload a media file", type=accepted_types)
else:
    pdf_files = st.file_uploader(
        "Upload one or more PDFs",
        type=accepted_types,
        accept_multiple_files=True
    )

# -----------------------------
# Handle Media (single file)
# -----------------------------
if media_file is not None and file_type.startswith("Media"):
    # Compute a stable ID for the uploaded media based on its content
    file_bytes = media_file.getbuffer()
    file_id = hashlib.sha256(file_bytes).hexdigest()

    # Only process if this media hasn't been added before
    if file_id not in st.session_state.doc_ids:
        # Save the upload once with a safe name
        try:
            file_path, safe_name = save_uploaded_file(media_file)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        st.session_state.processed_file = safe_name
        st.session_state.original_name = media_file.name

                # Transcribe
        content: Optional[str] = None
        if asr_model:
            content = transcribe_media_file_fast(
                asr_model,
                file_path,
                cache_key=file_id,
                lang=asr_language
            )
        else:
            st.error("ASR model not loaded. Cannot process media.")

        # If we got a transcript, add it to the shared vectorstore
        if content and embeddings and llm:
            # store raw text for initial global summary
            st.session_state.raw_docs[file_id] = content

            add_content_to_vectorstore(
                content=content,
                embeddings=embeddings,
                doc_id=file_id,
                source_name=media_file.name,
            )
            st.session_state.doc_ids.add(file_id)
            # Rebuild RAG chain over all current docs
            st.session_state.rag_chain = create_rag_chain(llm, embeddings)

        else:
            st.error("Processing media failed. Check model status and file content.")
            st.stop()
    # else: media already added → do nothing (reuse existing rag_chain)

# -----------------------------
# Handle Documents (multi-PDF)
# -----------------------------
if pdf_files and file_type.startswith("Document"):
    any_new_docs = False
    total_pdfs = len(pdf_files)
    progress = st.progress(0.0)

    with st.spinner("Processing and indexing uploaded PDFs..."):
        for idx, uploaded_file in enumerate(pdf_files, start=1):
            st.info(f"Processing {uploaded_file.name} ({idx}/{total_pdfs})")

            # Compute hash from file content
            file_bytes = uploaded_file.getbuffer()
            file_id = hashlib.sha256(file_bytes).hexdigest()

            # Skip PDFs already processed earlier
            if file_id in st.session_state.doc_ids:
                progress.progress(idx / total_pdfs)
                continue

            # Save safely
            try:
                file_path, safe_name = save_uploaded_file(uploaded_file)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            st.session_state.processed_file = safe_name
            st.session_state.original_name = uploaded_file.name

            # Load PDF text
            text = load_pdf_text_cached(file_path)
            content: Optional[str] = None

            if text and not text.startswith("__PDF_ERROR__::"):
                content = text
            elif text:
                err_msg = text.split("::", 1)[1]
                st.error(f"❌ Failed to load {uploaded_file.name}: {err_msg}")
            else:
                st.error(f"❌ Failed to load {uploaded_file.name}: Unknown error")

            # Add to vectorstore if OK
            if content and embeddings and llm:
                # store raw text for initial global summary
                st.session_state.raw_docs[file_id] = content

                add_content_to_vectorstore(
                    content=content,
                    embeddings=embeddings,
                    doc_id=file_id,
                    source_name=uploaded_file.name,
                )
                st.session_state.doc_ids.add(file_id)
                any_new_docs = True


            progress.progress(idx / total_pdfs)

    progress.empty()

    # If at least one new PDF was added, rebuild the RAG chain
    if any_new_docs:
        st.session_state.rag_chain = create_rag_chain(llm, embeddings)

# After handling media and/or documents, generate initial notes if needed
generate_initial_notes_if_needed()

# -----------------------------
# Q&A Section
# -----------------------------
st.markdown("---")
st.header("2) Ask Questions")

if st.session_state.rag_chain:
    st.success("Files are processed and ready for Q&A.")

    # Show prior messages (includes auto-generated notes at the top)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # New input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    placeholder = st.empty()
                    out_text = ""
                    for chunk in stream_answer(st.session_state.rag_chain, prompt):
                        out_text += chunk
                        placeholder.markdown(out_text)
                    response = out_text
                except Exception as e:
                    response = f"Error during generation: {e}"
                    placeholder.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Upload and process at least one file to begin the Q&A session.")
