# core/doc_utils.py
import os
import re
import uuid
from pathlib import Path
from typing import Optional, Any, List
from filelock import FileLock

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama  # only if used elsewhere, kept for typing-style clarity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# NOTE: this module expects the following names to exist in your environment or st.session_state:
# - CHROMA_DIR (string): persistent chroma directory (from your main.py config)
# - MAX_DOC_CHARS (int), INITIAL_SUMMARY_MAX_CHARS (int)
# They were originally module-level constants in main.py. If they are still defined in main.py,
# the module will use them via globals(); otherwise you can set them in st.session_state or env.

# If your main.py still defines module-level constants, they will be visible here if main imports this module
# after defining them. But to be robust we also attempt a few sensible fallbacks:


CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
MAX_DOC_CHARS = int(os.environ.get("MAX_DOC_CHARS", "200000"))
INITIAL_SUMMARY_MAX_CHARS = int(os.environ.get("INITIAL_SUMMARY_MAX_CHARS", "16000"))


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


auto_prompt_template = """
You are an expert study assistant. From the text given in [Context Start]...[Context End],
automatically DETECT the most important topics and produce a clean, structured study overview
that helps a student learn the subject from scratch.

[Context Start]
{combined_text}
[Context End]

You MUST produce output in **two clearly separated sections**, in Markdown, with the exact headings:

1) MAIN TOPICS
2) SUMMARY OF EACH TOPIC

SECTION 1 – MAIN TOPICS
- Print exactly this line on its own: MAIN TOPICS
- On the lines after that, output a bullet list of the 6–12 most important, high-level topics found in the materials.
- Each bullet:
  - MUST start with "- " (dash + space).
  - MUST be a short topic title of 2–6 words.
  - MUST NOT contain explanations, examples, or long sentences.
- One topic per bullet, one bullet per line.

SECTION 2 – SUMMARY OF EACH TOPIC
- After the last main topic bullet, leave a blank line.
- Then print exactly this line on its own: SUMMARY OF EACH TOPIC
- After that heading, for each main topic (in the same order as above), output ONE bullet:
  - The bullet MUST start with "- **Topic Name:**" followed by 2–4 full sentences.
  - These sentences must explain the topic from scratch for a beginner:
    - define the concept,
    - explain its purpose or role,
    - mention the most important sub-ideas or components mentioned in the text,
    - and, if helpful, give a simple example.
  - Each sentence MUST be a complete sentence in simple English, NOT just keywords or short fragments.
- Do NOT use nested bullets, "Subpoint", numbering (1), (2), etc., or any other list style in this section.
- There should be exactly one bullet per topic in the SUMMARY OF EACH TOPIC section.

GENERAL RULES
- Use ONLY markdown bullets starting with "- " as described.
- Do NOT invent topics or details that are not present in the text.
- If diagrams or images are referenced but not visible, you may mention that in a sentence (for example: "A diagram is referenced but its details are not available in the text.").
- Keep language clear, friendly, and suitable for a first-year student.
- Aim to make each topic's summary understandable even for someone seeing it for the first time.

Your output MUST follow this structure exactly, for example:

MAIN TOPICS
- Topic A
- Topic B
- Topic C

SUMMARY OF EACH TOPIC
- **Topic A:** [2–4 full sentences explaining Topic A].
- **Topic B:** [2–4 full sentences explaining Topic B].
- **Topic C:** [2–4 full sentences explaining Topic C].
"""


# ---- helpers (cleanup_upload_dir referenced at bottom) ----
def clamp_content(content: str) -> str:
    """
    Limit document length to avoid huge memory / embedding time.
    (Behavior preserved exactly from main.py)
    """
    if len(content) > MAX_DOC_CHARS:
        st.warning(f"Document too large, truncating to first {MAX_DOC_CHARS} characters for speed.")
        return content[:MAX_DOC_CHARS]
    return content


# session_id: keep same generation logic as before, but store/read in st.session_state
session_id = st.session_state.get("session_id")
if not session_id:
    session_id = uuid.uuid4().hex
    st.session_state.session_id = session_id


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
    Add a single document's content into the shared vectorstore.
    - content: raw text (already loaded from PDF or transcript)
    - doc_id: stable id for this document (e.g. SHA-256 hash)
    - source_name: original filename (for metadata/debug)
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
    """
    Create a RAG chain over *all* documents currently stored
    in the shared vectorstore.
    Call this after you've added at least one document via add_content_to_vectorstore.
    """
       
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def build_stratified_context(full_text: str, max_chars: int) -> str:
    """
    Take slices from the start, middle, and end of full_text so that
    the total length is <= max_chars. This ensures the model sees
    early, mid, and late parts of long documents instead of only the beginning.
    """
    if len(full_text) <= max_chars:
        return full_text

    # Split budget into three roughly equal parts
    part = max_chars // 3

    start = full_text[:part]

    mid_start = max(len(full_text) // 2 - part // 2, 0)
    mid = full_text[mid_start:mid_start + part]

    end = full_text[-part:]

    return start + "\n\n" + mid + "\n\n" + end


def normalize_bullets(text: str) -> str:
    """
    Enforce clean markdown bullets:
    - Replace common bullet glyphs with '- '.
    - Ensure each '- ' starts on its own line.
    - Strip 'Subpoint N:' labels.
    """
    if not text:
        return text

    # Strip "Subpoint 1:", "Subpoint 2:" etc
    text = re.sub(r'\bSubpoint\s*\d*\s*:\s*', '', text, flags=re.IGNORECASE)

    # Replace common bullet glyphs with newline + dash (top-level for now)
    for ch in ["•", "·", "◦", "▪"]:
        text = text.replace(ch, "\n- ")

    # Fix cases where the model did ", - " or "; - " inline
    text = text.replace(", - ", "\n- ")
    text = text.replace("; - ", "\n- ")

    # If the model put many bullets on one line, split them
    lines = text.splitlines()
    fixed_lines = []
    for ln in lines:
        # Count "- " occurrences; if more than one and line doesn't already start with "- "
        if ln.count("- ") > 1 and not ln.strip().startswith("- "):
            parts = ln.split("- ")
            for p in parts:
                p = p.strip()
                if p:
                    fixed_lines.append("- " + p)
        else:
            fixed_lines.append(ln)
    text = "\n".join(fixed_lines)

    # Collapse multiple blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


# core/doc_utils.py

def generate_initial_notes_from_text(
    combined_text: str,
    llm,
    max_chars: int = INITIAL_SUMMARY_MAX_CHARS,
) -> str:
    
    """
    Pure function: given combined document text and an LLM,
    return the initial notes string (MAIN TOPICS + SUMMARY OF EACH TOPIC).
    No Streamlit, no session_state, no side effects.
    """
    # reuse your existing helper
    combined_text = build_stratified_context(combined_text, max_chars)

    raw_notes = llm.invoke(auto_prompt_template.format(combined_text=combined_text))
    notes = normalize_bullets(raw_notes)

    # same sanity check as before
    if not notes.strip().upper().startswith("MAIN TOPICS"):
        notes = (
            "MAIN TOPICS\n- (could not extract topics cleanly)\n\n"
            "SUMMARY OF EACH TOPIC\n- (no summary generated)\n\n"
            + notes
        )

    return notes



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
