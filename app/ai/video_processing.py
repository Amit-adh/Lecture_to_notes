import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from moviepy.editor import VideoFileClip # type: ignore

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

processor = AutoProcessor.from_pretrained(model_id)

#settings changed a little for max util in collab
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    chunk_length_s=30,
    stride_length_s=(5, 5),
)

#load the video (mp4) and convert into audio (mp3)
video = VideoFileClip("test3.mp4")
video.audio.write_audiofile("test3-1.mp3")

#transcribing
result = pipe("test3-1.mp3", generate_kwargs={"return_timestamps": False, "num_beams": 1})

#summary of the transcription
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(result['text'], max_length=200, min_length=30, do_sample=False)
print(summary[0]['summary_text'])
