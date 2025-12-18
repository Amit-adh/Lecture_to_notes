# core/rag_chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Any

def default_prompt_template() -> ChatPromptTemplate:
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

    return ChatPromptTemplate.from_template(template)

def create_rag_chain(retriever: Any, llm: Any, prompt: ChatPromptTemplate = None):
    prompt = prompt or default_prompt_template()
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain



def generate_initial_notes_if_needed():
    """
    Generate an initial, structured overview (topic-agnostic) from uploaded documents.

    Output MUST be exactly:
    Main Topics:
    - Topic 1
    - Topic 2
    ...

    Topic-wise Summary:
    - **Topic 1:** one or two sentence overview.
      - sub-point 1 (one sentence)
      - sub-point 2 (one sentence)
    - **Topic 2:** one or two sentence overview.
      - ...
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
    combined_text = build_stratified_context(combined_text, INITIAL_SUMMARY_MAX_CHARS)

    # Topic-agnostic prompt that asks the model to detect important topics and subtopics automatically.
    auto_prompt = f"""
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


    try:
        with st.spinner("Generating main topics and topic-wise summary from the uploaded materials..."):
            raw_notes = llm.invoke(auto_prompt)
            notes = normalize_bullets(raw_notes)
    except Exception as e:
        st.error(f"Error while generating initial notes: {e}")
        return

    # Simple sanity check / minimal reformat
        # Simple sanity check / minimal reformat
    if not notes.strip().upper().startswith("MAIN TOPICS"):
        notes = "MAIN TOPICS\n- (could not extract topics cleanly)\n\nSUMMARY OF EACH TOPIC\n- (no summary generated)\n\n" + notes

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
        if hasattr(chain, "stream"):
            for chunk in chain.stream(question):
                yield str(chunk)
        else:
            yield chain.invoke(question)
    except Exception as e:
        # Surface a readable error instead of crashing the whole app
        yield f"\n[Error during generation: {e}]"
