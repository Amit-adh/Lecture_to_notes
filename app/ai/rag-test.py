import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import requests

# --- STEP 1: Whisper Transcription ---
print("Loading model...")

device = 0 if torch.cuda.is_available() else -1


model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16, #if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True
    #force_download=True  # force clean re-download
)

print("Loading processor...")

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

print("Model and processor loaded successfully.")


pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device=device,
    chunk_length_s=30,
    stride_length_s=(5, 5),
)

result = pipe("C:/Users/Dell/Downloads/test2.mp3", generate_kwargs={"return_timestamps": False, "num_beams": 1})
transcription = result["text"]

print("TRANSCRIPT:")
print(transcription)

# --- STEP 2: Summarization via Ollama ---
# Make sure ollama is running and model (e.g., llama3) is pulled via: `ollama run llama3`

transcription = " You know, it's actually really sad. Let me see if I can show you guys. This is what it was. Just watch on kick. I don't know if I get in trouble. It wouldn't make any sense to me. But yeah, anyway, so it could be pilot error. Yeah. And you see that plane coming up, right? You see it going up. The one guy survived, barely sustained any injuries. Everyone else died. One dude actually survived this. God damn. And you can see like right there, it's starting to lose. Seems like the engine engine like it was like things were going and then they stopped going right something cut out yeah it was like an engine stall or something like that losing lift well i feel like you don't lose lift like that do you because like i feel like if you would lose lift like that well i don't know i mean what what the fuck do i know right and then you can you can see, obviously, it blew up. Oh, my God. How did that dude survive, bro? That's like some Bruce Willis unbreakable shit. Oh, my God. So you're going to have to wait and find out. See if it was pilot error. See if it was an engine failure. How do engine failures like that happen, right? I mean, is it a defective part? Is it something that was done at at manufacturing was there a safety protocol that wasn't followed uh there's so many different examples of things that it could be and it's hard to know which one it is right i mean i would usually guess that it's pilot error and it's not some kind of like crazy weird conspiracy but who the fuck knows so yeah"
ollama_response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:1b",  # Or another summarization-capable model
        "prompt": f"Summarize the following transcription:\n\n{transcription}",
        "stream": False
    }
)
summary = ""
print(transcription)

if ollama_response.ok:
    summary = ollama_response.json().get("response", "").strip()
    print("\nSUMMARY:")
    print(summary)
else:
    print("Error:", ollama_response.status_code, ollama_response.text)

ollama_response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:1b",  # Or another summarization-capable model
        "prompt": f"What is the topic of this text:\n\n{summary}",
        "stream": False
    }
)

if ollama_response.ok:
    summary = ollama_response.json().get("response", "").strip()
    print("\nQUESTION: What is the topic of this text?\n\nANSWER:")
    print(summary)
else:
    print("Error:", ollama_response.status_code, ollama_response.text)

