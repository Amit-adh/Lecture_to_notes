
import os
import uuid
import streamlit as st
import torch
import hashlib
from typing import Optional

# --- LLM / RAG ---
from transformers import logging as hf_logging  # quiet HF logs
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import *


# for the api key
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ.setdefault("OMP_NUM_THREADS", str(MAX_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(MAX_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(MAX_THREADS))


# -----------------------------
# Global Configuration
# -----------------------------

st.set_page_config(page_title="AI Document & Media Q&A", layout="wide")

# ASR model sizing (adjust for your machine)
ASR_MODEL_SIZE = os.environ.get("ASR_MODEL_SIZE", "small")  # "distil-medium.en" | "base" | "small" | "medium"
ASR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_COMPUTE = os.environ.get("ASR_COMPUTE", "float16" if ASR_DEVICE == "cuda" else "int8")

# Ensure dirs
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_CACHE_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Quiet down HF warnings
hf_logging.set_verbosity_error()


# -----------------------------
# Helpers
# -----------------------------

from core.file_utils import (
    cleanup_upload_dir,
    save_uploaded_file,
    cached_transcript_path,
    normalize_to_wav16k,
)


from core.doc_utils import (
    clamp_content,
    get_vectorstore_pure,
    add_content_to_vectorstore_pure,
    create_rag_chain_pure,
    generate_initial_notes_from_text,
    stream_answer,
)

# -----------------------------
# Cached Resources
# -----------------------------


# --- needs slight change ---
@st.cache_resource
def load_groq_client():
    return Groq(api_key=groq_api_key)


@st.cache_resource
def load_groq_llm():
    st.write("Cache miss: initializing Groq LLM...")
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            groq_api_key=groq_api_key,
        )

        # Warmup (tiny)
        _ = llm.invoke("hello")

        st.write("Groq LLM ready.")
        return llm

    except Exception as e:
        st.error(f"Groq LLM init failed. Error: {e}")
        return None


@st.cache_resource
def load_embeddings():
    st.write("Cache miss: loading HF embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    st.write("HF embeddings ready.")
    return embeddings


# -----------------------------
# Cached Data Helpers
# -----------------------------

from core.pdf_utils import (
    chunk_text_to_docs,
    load_pdf_text,
)


@st.cache_data(show_spinner=False)
def split_text_cached(content: str):
    return chunk_text_to_docs(content)


@st.cache_data
def load_pdf_text_cached(file_path: str) -> Optional[str]:
    return load_pdf_text(file_path)


# -----------------------------
# Core Processing
# -----------------------------

def transcribe_media_file(
    client, 
    file_path: str, 
    cache_key: str, 
    lang: Optional[str]
) -> Optional[str]:
    
    """
    Transcribe media using Groq Whisper API.
    Caches transcript by file hash.
    """
    cache_path = cached_transcript_path(cache_key, TRANSCRIPT_CACHE_DIR)

    if os.path.exists(cache_path):
        st.info("Using cached transcript.")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    st.info("Starting transcription...")

    try:
        wav_path = normalize_to_wav16k(file_path, cache_key, UPLOAD_DIR)
    except Exception as e:
        st.error(f"Audio preprocessing failed: {e}")
        return None
    
   
    final_text: Optional[str]
    
    with st.spinner("Transcribing audio..."):
        # Call Groq Whisper
        try:
            with open(wav_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    language=lang,   # None = auto-detect
                )
            final_text = transcription.text.strip()

        except Exception as e:
            st.error(f"Transcription error: {e}")
            return None

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    st.success("Transcription complete.")
    return final_text


# cached resource for vectorstore (Streamlit)
@st.cache_resource
def get_vectorstore_cached(_embeddings, session_id):
    return get_vectorstore_pure(_embeddings, session_id, CHROMA_DIR)

# When adding content (use split_text_cached from UI)
def add_content_to_vectorstore(
    content: str,
    embeddings,
    doc_id: str,
    source_name: str,
    split_fn,
):
    content = clamp_content(content)
    vs = get_vectorstore_cached(embeddings, st.session_state.session_id)
    # Wrap with spinner and error UI
    with st.spinner("Indexing document into vectorstore..."):
        add_content_to_vectorstore_pure(
            content=content,
            embeddings=embeddings,
            doc_id=doc_id,
            source_name=source_name,
            vs=vs,
            split_fn=split_fn,
            chroma_dir=CHROMA_DIR,
        )

# Adapter to create rag_chain (returns chain to use in streaming)
def create_rag_chain(llm, embeddings):
    vs = get_vectorstore_cached(embeddings, st.session_state.session_id)
    with st.spinner("Building retriever and RAG chain from the uploaded materials..."):
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "fetch_k": 120, "lambda_mult": 0.25},
        )
        return create_rag_chain_pure(retriever, llm)


def generate_initial_notes_if_needed(
    llm, 
    embeddings,
    retriever_k=50, 
    fetch_k=200, 
    max_chars=INITIAL_SUMMARY_MAX_CHARS,
):
    if not llm or not embeddings:
        return
    if not st.session_state.get("doc_ids"):
        return

    # detect doc set changed
    current_state = frozenset(st.session_state.doc_ids)
    if current_state == st.session_state.get("last_notes_doc_state"):
        return

    vs = get_vectorstore_cached(embeddings, st.session_state.session_id)
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": retriever_k,
            "fetch_k": fetch_k,
            "lambda_mult": 0.25,
        },
    )

    # retrieve representative docs (use retriever or fallback)
    try:
        docs = retriever.get_relevant_documents("")
    except Exception:
        docs = vs.similarity_search("", k=retriever_k)

    if not docs:
        return

    combined_text = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    # call LLM to produce notes 
    with st.spinner("Generating main topics and topic-wise summary from the uploaded materials..."):
        notes = generate_initial_notes_from_text(
            combined_text=combined_text,
            llm=llm,
            max_chars=max_chars,
        )

    st.session_state.messages.append({"role": "assistant", "content": notes})
    st.session_state.last_notes_doc_state = current_state


# Run this once per server process
cleanup_upload_dir(UPLOAD_DIR)


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
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex



# -----------------------------
# Sidebar: Model Status
# -----------------------------
with st.sidebar:
    st.header("Models & Settings")
    with st.status("Initializing...", expanded=False) as status:
        asr_model = load_groq_client()     
        llm = load_groq_llm()          
        embeddings = load_embeddings()    # HF embeddings

        st.session_state["llm"] = llm
        st.session_state["embeddings"] = embeddings

        st.session_state["llm"] = llm
        st.session_state["embeddings"] = embeddings

        if all([asr_model, llm, embeddings]):
            status.update(label="Models Ready", state="complete", expanded=False)
        else:
            status.update(label="Model init failed", state="error", expanded=True)
            st.error("Please ensure Ollama is running and models exist (ollama pull ...).")

    st.subheader("ASR Language")
    asr_language = st.selectbox(
        "Pick language (Auto is fine if unsure):",
        options=["Auto", "en", "hi", "es", "fr", "de", "zh", "ja"],
        index=0
    )
    asr_language = None if asr_language == "Auto" else asr_language

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
st.markdown("Upload a media file (`mp3`, `wav`, `mp4`, `mov`) or one/more documents (`pdf`, `pptx`) and ask questions about their content.")

st.header("1) Choose File Type")
file_type = st.radio("Select file type:", ("Media (Audio/Video)", "Document (PDF/pptx)"), horizontal=True)

accepted_types = ["mp3", "wav", "mp4", "mov"] if file_type.startswith("Media") else ["pdf", "pptx"]

media_file = None
pdf_files = None

if file_type.startswith("Media"):
    media_file = st.file_uploader("Upload a media file", type=accepted_types)
else:
    pdf_files = st.file_uploader(
        "Upload one or more PDFs or pptx",
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
            file_path, safe_name = save_uploaded_file(media_file, UPLOAD_DIR, MAX_FILE_MB)
            
        except ValueError as e:
            st.error(str(e))
            st.stop()

        st.session_state.processed_file = safe_name
        st.session_state.original_name = media_file.name

        # Transcribe
        content: Optional[str] = None
        if asr_model:
            content = transcribe_media_file(
                client=asr_model,
                file_path=file_path,
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
                split_fn=split_text_cached,
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
                file_path, safe_name = save_uploaded_file(uploaded_file, UPLOAD_DIR, MAX_FILE_MB)
                
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
                    split_fn=split_text_cached
                )
                st.session_state.doc_ids.add(file_id)
                any_new_docs = True


            progress.progress(idx / total_pdfs)

    progress.empty()

    # If at least one new PDF was added, rebuild the RAG chain
    if any_new_docs:
        st.session_state.rag_chain = create_rag_chain(llm, embeddings)

# After handling media and/or documents, generate initial notes if needed
generate_initial_notes_if_needed(llm=llm, embeddings=embeddings)

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
