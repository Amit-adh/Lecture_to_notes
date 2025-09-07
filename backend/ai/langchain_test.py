import streamlit as st
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
# To handle video files, you'll need moviepy
# pip install moviepy
from moviepy.editor import VideoFileClip

# --- GLOBAL CONFIGURATION ---
UPLOAD_DIRECTORY = "uploads"
OLLAMA_MODEL = "llama3" # More standard model name, change if needed

# Create the upload directory if it doesn't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# --- MODEL & PIPELINE LOADING (CACHED) ---
# This section uses Streamlit's caching to load models only once.

@st.cache_resource
def load_speech_recognition_pipeline():
    """Loads and caches the Whisper model and pipeline."""
    st.write("Cache miss: Loading speech recognition model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
    )
    st.write("Speech recognition model loaded.")
    return pipe

@st.cache_resource
def load_ollama_llm_and_embeddings():
    """Loads and caches the Ollama LLM and embeddings models."""
    st.write(f"Cache miss: Initializing Ollama with model '{OLLAMA_MODEL}'...")
    llm = Ollama(model=OLLAMA_MODEL)
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    # A simple check to see if the model is available.
    try:
        llm.invoke("Hi")
        st.write("Ollama models are ready.")
        return llm, embeddings
    except Exception as e:
        st.error(f"Ollama connection failed. Is Ollama running? Error: {e}")
        return None, None

# --- BACKEND PROCESSING FUNCTIONS ---
# These functions contain the logic from your script.

def transcribe_media_file(pipe, file_path):
    """
    Transcribes an audio or video file using the pre-loaded Whisper pipeline.
    If it's a video, it extracts audio first.
    """
    try:
        # Handle video files by extracting audio
        if file_path.lower().endswith('.mp4'):
            st.info("Video file detected. Extracting audio...")
            video = VideoFileClip(file_path)
            audio_path = os.path.join(UPLOAD_DIRECTORY, "temp_audio.mp3")
            video.audio.write_audiofile(audio_path)
            file_path = audio_path # Use the extracted audio for transcription

        with st.spinner(f"Transcribing audio file... This may take a moment."):
            with open(file_path, "rb") as f:
                result = pipe(f.read(), generate_kwargs={"language": "english"})
        
        transcription = result["text"].strip()
        return transcription

    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        return None

def process_transcription_with_rag(transcription, llm, embeddings):
    """
    Takes a transcription, creates a vector store, and performs RAG-based
    summarization, topic extraction, and quiz generation.
    """
    if not transcription:
        return None, None, None

    with st.spinner("Splitting text and creating vector store..."):
        # 1. Split the transcription into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.create_documents([transcription])

        # 2. Create a vector store
        vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
        retriever = vectorstore.as_retriever()

    with st.spinner("Performing RAG-based analysis..."):
        # 3. Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # 4. Get Summary
        summary_query = "Summarize the provided text in a few concise sentences."
        summary_result = qa_chain.invoke({"query": summary_query})
        summary = summary_result.get("result", "").strip()

        # 5. Get Topic
        topic_prompt = f"Based on the following text, what is the main topic? Answer in under 10 words.\n\n{summary}"
        topic = llm.invoke(topic_prompt).strip()

        # 6. Get Quiz Question
        quiz_prompt = f"Based on the following text, generate one multiple-choice quiz question to test a student's understanding. Provide the question, options (A, B, C, D), and the correct answer.\n\n{summary}"
        quiz = llm.invoke(quiz_prompt).strip()

    return summary, topic, quiz

# --- STREAMLIT FRONTEND ---
st.set_page_config(page_title="AI Media Analyzer", layout="wide")
st.title("🤖 AI Media File Analyzer")
st.markdown("Upload an audio (`.mp3`, `.wav`) or video (`.mp4`) file. The system will transcribe it, then use a RAG pipeline to summarize the content, identify the main topic, and generate a quiz question.")

# Load models on startup and show status
with st.status("🚀 Initializing AI models...", expanded=True) as status:
    speech_pipe = load_speech_recognition_pipeline()
    llm, embeddings = load_ollama_llm_and_embeddings()
    if speech_pipe and llm and embeddings:
        status.update(label="✅ AI Models are ready!", state="complete", expanded=False)
    else:
        status.update(label="⚠️ Model loading failed. Check logs.", state="error", expanded=True)

# File Uploader
media_file = st.file_uploader(
    "Choose a media file",
    type=["mp3", "wav", "mp4"]
)

if media_file is not None:
    # Save the uploaded file locally
    file_path = os.path.join(UPLOAD_DIRECTORY, media_file.name)
    with open(file_path, "wb") as f:
        f.write(media_file.getbuffer())

    # Display the uploaded media
    st.markdown("---")
    file_type = media_file.type
    if 'audio' in file_type:
        st.audio(file_path)
    elif 'video' in file_type:
        st.video(file_path)

    # Process button
    if st.button("Analyze Media File", type="primary", use_container_width=True):
        if not speech_pipe or not llm:
            st.error("Models are not available. Cannot process the file.")
        else:
            # Step 1: Transcription
            transcription = transcribe_media_file(speech_pipe, file_path)
            
            if transcription:
                st.subheader("📝 Full Transcription")
                st.text_area("Transcription", transcription, height=250)

                # Step 2: RAG Processing
                summary, topic, quiz = process_transcription_with_rag(transcription, llm, embeddings)
                
                if summary:
                    st.subheader("🧠 RAG-Based Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Main Topic:**\n\n{topic}")
                    with col2:
                        st.success(f"**Summary:**\n\n{summary}")
                    
                    st.markdown("**Generated Quiz Question:**")
                    st.code(quiz, language=None)

