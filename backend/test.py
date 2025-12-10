import streamlit as st
import os
import torch
import ffmpeg
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- GLOBAL CONFIGURATION ---
UPLOAD_DIRECTORY = "uploads"
OLLAMA_LLM_MODEL = "koesn/llama3-8b-instruct:latest" 
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text" # Using a dedicated embedding model

# Ensure upload directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# --- MODEL & PIPELINE LOADING (CACHED) ---

@st.cache_resource
def load_speech_recognition_pipeline():
    """Load and cache the Whisper speech recognition model."""
    st.write("Cache miss: Loading speech recognition model (Whisper)...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    whisper_model = "openai/whisper-large-v3" if torch.cuda.is_available() else "openai/whisper-base"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_model, torch_dtype=torch_dtype, use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(whisper_model)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if torch.cuda.is_available() else -1,
        return_timestamps=True
    )
    st.write(f"✅ Whisper model '{whisper_model}' loaded on {device}.")
    return pipe

@st.cache_resource
def load_ollama_llm():
    """Load and cache the Ollama LLM."""
    st.write(f"Cache miss: Initializing Ollama LLM '{OLLAMA_LLM_MODEL}'...")
    try:
        llm = Ollama(model=OLLAMA_LLM_MODEL)
        llm.invoke("Hi") # Test connection
        st.write("✅ Ollama LLM is ready.")
        return llm
    except Exception as e:
        st.error(f"Ollama LLM connection failed. Is Ollama running? Error: {e}")
        return None

@st.cache_resource
def load_ollama_embeddings():
    """Load and cache the Ollama embeddings."""
    st.write(f"Cache miss: Initializing Ollama embeddings '{OLLAMA_EMBEDDING_MODEL}'...")
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        st.write("✅ Ollama embeddings are ready.")
        return embeddings
    except Exception as e:
        st.error(f"Ollama embeddings connection failed. Is Ollama running? Error: {e}")
        return None

# --- FILE PROCESSING LOGIC ---

def transcribe_media_file(pipe, file_path):
    """Extracts audio if necessary and transcribes it."""
    st.info("File detected. Starting transcription process...")
    try:
        # Check for video and extract audio
        if file_path.lower().endswith(('.mp4', '.mov', '.avi')):
            audio_path = os.path.join(UPLOAD_DIRECTORY, "temp_audio.mp3")
            (
                ffmpeg.input(file_path)
                .output(audio_path, format="mp3", acodec="libmp3lame")
                .overwrite_output()
                .run(quiet=True)
            )
            file_to_transcribe = audio_path
        else:
            file_to_transcribe = file_path
        
        with st.spinner("⏳ Transcribing... this may take a moment."):
            result = pipe(file_to_transcribe)
        st.success("Transcription complete.")
        return result["text"].strip()
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

def load_pdf_text(file_path):
    """Loads and extracts text from a PDF file."""
    st.info("PDF detected. Loading content...")
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        st.success("PDF content loaded.")
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        st.error(f"Failed to load PDF: {e}")
        return None

# --- RAG CHAIN CREATION ---

def create_rag_chain(content, llm, embeddings):
    """Creates a RAG chain from text content."""
    if not content:
        st.warning("Cannot create RAG chain from empty content.")
        return None
    
    with st.spinner("🧠 Processing content for Q&A..."):
        # 1. Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.create_documents([content])

        # 2. Create vector store from chunks
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

        # 3. Create retriever
        retriever = vectorstore.as_retriever()
        
        # 4. Define prompt template
        prompt_template = """[INST] You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Keep the answer concise and helpful.
        
        Context: {context}
        
        Question: {question} 
        
        Answer: [/INST]"""
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # 5. Create RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    return rag_chain

# --- STREAMLIT UI ---

st.set_page_config(page_title="AI Document & Media Q&A", layout="wide")
st.title("AI Document & Media Q&A 💬")
st.markdown("Upload a media file (`mp3`, `wav`, `mp4`) or a document (`pdf`) to ask questions about its content.")

# Initialize session state variables
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# Load models on startup
with st.sidebar:
    st.header("Model Status")
    with st.status("Initializing models...", expanded=True) as status:
        speech_pipe = load_speech_recognition_pipeline()
        llm = load_ollama_llm()
        embeddings = load_ollama_embeddings()
        
        if all([speech_pipe, llm, embeddings]):
            status.update(label="✅ Models Ready!", state="complete", expanded=False)
        else:
            status.update(label="⚠️ Model loading failed.", state="error", expanded=True)
            st.error("Please ensure Ollama is running and all models are downloaded.")

# --- File Upload and Processing ---

st.header("1. Choose Your File Type")
file_type = st.radio("Select file type:", ("Media (Audio/Video)", "Document (PDF)"), horizontal=True)

if file_type == "Media (Audio/Video)":
    uploaded_file = st.file_uploader("Upload a media file", type=["mp3", "wav", "mp4", "mov"])
else:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Process the file only if it's new
    if uploaded_file.name != st.session_state.processed_file:
        st.session_state.rag_chain = None # Reset chain for new file
        st.session_state.messages = [] # Reset chat
        
        file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        content = None
        if file_type == "Media (Audio/Video)":
            if speech_pipe:
                content = transcribe_media_file(speech_pipe, file_path)
            else:
                st.error("Speech recognition model not loaded. Cannot process media.")
        else: # PDF
            content = load_pdf_text(file_path)

        if content and llm and embeddings:
            st.session_state.rag_chain = create_rag_chain(content, llm, embeddings)
            st.session_state.processed_file = uploaded_file.name
        else:
            st.error("Processing failed. Please check model status and file content.")
    
# --- Q&A Chat Interface ---

st.markdown("---")
st.header("2. Ask Questions About Your File")

if st.session_state.rag_chain:
    st.success(f"✅ Ready to answer questions about **{st.session_state.processed_file}**")
    
    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get new user input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(prompt)
                st.markdown(response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Upload and process a file to begin the Q&A session.")