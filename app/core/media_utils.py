# core/media_utils.py
import os
import subprocess
from typing import Optional
from faster_whisper import WhisperModel

def ffmpeg_to_wav16k_mono(src: str, dst: str):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", src, "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-f", "wav", dst]
    subprocess.run(cmd, check=True, timeout=300)

def normalize_to_wav16k(src: str, dst: str):
    if os.path.exists(dst):
        return dst
    ffmpeg_to_wav16k_mono(src, dst)
    return dst

def load_asr_model(size="small", device="cpu", compute_type=None) -> WhisperModel:
    compute = compute_type or ("float16" if device == "cuda" else "int8")
    return WhisperModel(size, device=device, compute_type=compute)

def transcribe_wav(asr_model: WhisperModel, wav_path: str, lang: Optional[str] = None) -> Optional[str]:
    try:
        kwargs = dict(vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
                      beam_size=1, best_of=1, temperature=0, task="transcribe")
        if lang and lang.lower() != "auto":
            kwargs["language"] = lang
        segments, info = asr_model.transcribe(wav_path, **kwargs)
        return " ".join(seg.text for seg in segments).strip()
    except Exception:
        return None
