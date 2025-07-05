import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import requests

# # --- STEP 1: Whisper Transcription ---
# print("Loading model...")

# device = 0 if torch.cuda.is_available() else -1


# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     "openai/whisper-large-v3",
#     torch_dtype=torch.float16, #if torch.cuda.is_available() else torch.float32,
#     low_cpu_mem_usage=True,
#     use_safetensors=True
#     #force_download=True  # force clean re-download
# )

# print("Loading processor...")

# processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

# print("Model and processor loaded successfully.")


# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     torch_dtype=torch.float16,
#     device=device,
#     chunk_length_s=30,
#     stride_length_s=(5, 5),
# )

# result = pipe("C:/Users/Dell/Downloads/test2.mp3", generate_kwargs={"return_timestamps": False, "num_beams": 1})
# transcription = result["text"]

# print("TRANSCRIPT:")
# print(transcription)

# --- STEP 2: Summarization via Ollama ---
# Make sure ollama is running and model (e.g., llama3) is pulled via: `ollama run llama3`

transcription = "Okay, so I was just wondering what are the capabilities of this specific, let's say program, or should I say program? Should I not say program? I don't really know, but let's say a feature of this model. Model would be a good word. So I just want to test how good this model really is, and I want to try with a lecture. So let's try. The first thing is going to be, we're going to be talking about AI models. This is a test for the whispers large v3 model by open AI and you can get it in hugging face now this audio recording is already over 30 seconds and I want to see how long does it take for the audio recording to get processed the five second one that I tested earlier only took four seconds so does it scale up with linear like with time? Or is AI smart enough to break down certain long-scale recordings into smaller sections and process them parallelly? Let's see."

ollama_response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:1b",  # Or another summarization-capable model
        "prompt": f"Summarize the following transcription:\n\n{transcription}",
        "stream": False
    }
)

if ollama_response.ok:
    summary = ollama_response.json().get("response", "").strip()
    print("\nSUMMARY:")
    print(summary)
else:
    print("Error:", ollama_response.status_code, ollama_response.text)
