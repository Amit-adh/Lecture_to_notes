import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import requests
import sys # Used for clean exit
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# --- SETUP ---
AUDIO_FILE_PATH = "C:/Users/Dell/Downloads/test2.mp3"
OLLAMA_MODEL = "koesn/llama3-8b-instruct:latest"


# --- STEP 1: Whisper Transcription ---
print("Loading speech recognition model...")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    print("Model and processor loaded successfully.")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
    )

    print(f"\nTranscribing audio file: {AUDIO_FILE_PATH}...")
    with open(AUDIO_FILE_PATH, "rb") as f:
        result = pipe(f.read(), generate_kwargs={"language": "english", "num_beams": 1})

    transcription = result["text"].strip()

    print("\n✅ TRANSCRIPTION COMPLETE:")
    print(transcription)

except FileNotFoundError:
    print(f"❌ ERROR: Audio file not found at '{AUDIO_FILE_PATH}'. Please check the path.")
    sys.exit() # Exit the script if the file isn't found
except Exception as e:
    print(f"❌ An error occurred during transcription: {e}")
    sys.exit()


# --- STEP 2: Summarization & Topic Extraction via Ollama (RAG-based) ---

# 1. Initialize Ollama LLM and Embeddings
llm = Ollama(model=OLLAMA_MODEL)
embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

# 2. Split the transcription into chunks
print("\nSplitting transcription into chunks for RAG...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, # Adjust chunk size as needed
    chunk_overlap=50 # Adjust overlap as needed
)
texts = text_splitter.create_documents([transcription])
print(f"Created {len(texts)} chunks.")

# 3. Create a vector store from the chunks
print("Creating vector store...")
# You can persist this to disk if you want to reuse it
vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
retriever = vectorstore.as_retriever()
print("Vector store created.")

# 4. Create a RetrievalQA chain for summarization
print("\nPerforming RAG-based summarization...")
qa_chain_summarize = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 'stuff' combines all retrieved documents into one prompt
    retriever=retriever,
    return_source_documents=True # Optional: to see which chunks were used
)

summary_query = "Summarize the provided text in a few sentences."
try:
    summary_result = qa_chain_summarize.invoke({"query": summary_query})
    summary = summary_result.get("result", "").strip()

    if summary:
        print("\n✅ RAG-BASED SUMMARY COMPLETE:")
        print(summary)
        
        # Optional: print source documents
        # print("\nSource documents used for summary:")
        # for doc in summary_result.get("source_documents", []):
        #     print(f"- {doc.page_content[:100]}...")
        
        print("\n--- TOP CHUNKS USED FOR SUMMARY ---")
        source_documents = summary_result.get("source_documents", [])
        if source_documents:
            for i, doc in enumerate(source_documents[:3]):
                print(f"\nChunk {i+1}:")
                print(doc.page_content)
                # You can also print metadata if available
                # if doc.metadata:
                #     print(f"  Metadata: {doc.metadata}")
        else:
            print("No source documents were returned for the summary. This might happen if the chain_type doesn't pass them, or if the retriever didn't find relevant chunks for the query.")
        print("-----------------------------------")

    # 5. Create a RetrievalQA chain for topic extraction (using the summary as context)
        print("\nPerforming RAG-based topic extraction...")
        # For topic extraction, we can directly query the LLM with the summary
        # or create a separate RAG chain if the topic needs to be derived directly from the original transcription chunks.
        # For simplicity and leveraging the already generated summary, we'll just query the LLM.
        
        topic_prompt = f"Based on the following text, what is the main topic? Answer in under 10 words.\n\n{summary}"
        
        # We'll use the direct LLM call here since the context is already summarized
        response = llm.invoke(topic_prompt)
        topic = response.strip()

        if topic:
            print("\n✅ RAG-BASED TOPIC:")
            print(topic)
    else:
        print("❌ Could not generate summary.")

except Exception as e:
    print(f"❌ An error occurred during RAG summarization or topic extraction: {e}")