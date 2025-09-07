import streamlit as st
import os
import torch
import ffmpeg
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# --- GLOBAL CONFIGURATION ---
UPLOAD_DIRECTORY = "uploads"
OLLAMA_MODEL = "koesn/llama3-8b-instruct:latest"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# --- MODEL & PIPELINE LOADING ---
@st.cache_resource
def load_speech_recognition_pipeline():
    """Load Whisper."""
    st.write("Cache miss: Loading speech recognition model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Choose smaller model if no GPU (large model too slow on CPU)
    whisper_model = "openai/whisper-small" if device == "cpu" else "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(whisper_model)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if torch.cuda.is_available() else -1,  # HuggingFace expects -1 for CPU
        return_timestamps=True,
    )
    st.write(f"Whisper model '{whisper_model}' loaded on {device}.")
    return pipe

@st.cache_resource
def load_ollama_llm_and_embeddings():
    """Load Ollama LLM + embeddings."""
    st.write(f"Cache miss: Initializing Ollama with model '{OLLAMA_MODEL}'...")
    llm = Ollama(model=OLLAMA_MODEL)
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    try:
        llm.invoke("Hi")
        st.write("Ollama models are ready.")
        return llm, embeddings
    except Exception as e:
        st.error(f"Ollama connection failed. Is Ollama running? Error: {e}")
        return None, None

# --- AUDIO EXTRACTION & TRANSCRIPTION ---
def extract_audio(file_path):
    """Extract audio from video if needed (optimized with ffmpeg)."""
    if file_path.lower().endswith(".mp4"):
        # st.info("Video detected. Extracting audio with ffmpeg...")
        audio_path = os.path.join(UPLOAD_DIRECTORY, "temp_audio.mp3")

        # Fast copy if audio already exists in MP4 (no re-encoding)
        try:
            (
                ffmpeg
                .input(file_path)
                .output(audio_path, codec="copy", vn=None)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error:
            # fallback to re-encoding if direct copy fails
            (
                ffmpeg
                .input(file_path)
                .output(audio_path, format="mp3", acodec="libmp3lame")
                .overwrite_output()
                .run(quiet=True)
            )
        return audio_path
    return file_path

def transcribe_media_file(pipe, file_path):
    """Run Whisper transcription on audio/video."""
    try:
        audio_file = extract_audio(file_path)
        with st.spinner("Transcribing audio..."):
            result = pipe(audio_file, generate_kwargs={"language": "english"})
        return result["text"].strip()
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

# --- RAG PROCESSING ---
def process_transcription_with_rag(transcription, llm, embeddings):
    if not transcription:
        return None, None, None

    with st.spinner("Processing..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.create_documents([transcription])
        vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
        retriever = vectorstore.as_retriever()

    with st.spinner("Performing RAG analysis..."):
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        summary_result = qa_chain.invoke({"query": "Summarize in 3-4 sentences."})
        summary = summary_result.get("result", "").strip()

        # topic = llm.invoke(
        #     f"Identify the main topic of this text in under 10 words:\n\n{summary}"
        # ).strip()

        # quiz = llm.invoke(
        #     f"Create one multiple-choice question (A–D) with the correct answer:\n\n{summary}"
        # ).strip()

    return summary, topic, quiz

# --- STREAMLIT FRONTEND ---
st.set_page_config(page_title="AI Media Analyzer", layout="wide")
st.title("AI Media File Analyzer")
st.markdown("Upload an audio (`.mp3`, `.wav`) or video (`.mp4`) file.")

with st.status("Initializing models...", expanded=True) as status:
    speech_pipe = load_speech_recognition_pipeline()
    llm, embeddings = load_ollama_llm_and_embeddings()
    if speech_pipe and llm and embeddings:
        status.update(label="Models ready!", state="complete", expanded=False)
    else:
        status.update(label="Model loading failed.", state="error", expanded=True)

media_file = st.file_uploader("Choose a media file", type=["mp3", "wav", "mp4"])

if media_file:
    file_path = os.path.join(UPLOAD_DIRECTORY, media_file.name)
    with open(file_path, "wb") as f:
        f.write(media_file.getbuffer())

    st.markdown("---")
    if 'audio' in media_file.type:
        st.audio(file_path)
    elif 'video' in media_file.type:
        st.video(file_path)

    if st.button("Analyze Media File", type="primary", use_container_width=True):
        if not speech_pipe or not llm:
            st.error("Models not ready. Cannot process file.")
        else:
            transcription = transcribe_media_file(speech_pipe, file_path)
            if transcription:
                st.subheader("Transcription")
                st.text_area("Full Text", transcription, height=250)

                summary, topic, quiz = process_transcription_with_rag(transcription, llm, embeddings)
                if summary:
                    st.subheader("Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Main Topic:** {topic}")
                    with col2:
                        st.success(f"**Summary:**\n\n{summary}")

                    st.markdown("**Generated Quiz:**")
                    st.code(quiz)
