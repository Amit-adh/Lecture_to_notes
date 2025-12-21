# core/media_utils.py

import os
import subprocess
from typing import Optional

def cached_transcript_path(content_hash: str, transcript_cache_dir: str) -> str:
    return os.path.join(transcript_cache_dir, f"{content_hash}.txt")


# def normalized_wav_cache_path(content_hash: str, upload_dir: str) -> str:
#     return os.path.join(upload_dir, f"{content_hash}_16k.wav")


# def ffmpeg_to_wav16k_mono(src: str, dst: str, ffmpeg_bin: str = "ffmpeg", timeout: int = 300):
#     """
#     Convert audio/video at `src` to 16kHz mono WAV at `dst` using ffmpeg.
#     Preserves original exceptions (TimeoutExpired -> RuntimeError, CalledProcessError -> RuntimeError).
#     """
#     cmd = [
#         ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
#         "-i", src, "-vn", "-sn", "-dn",
#         "-ac", "1", "-ar", "16000", "-f", "wav", dst
#     ]
#     try:
#         subprocess.run(cmd, check=True, timeout=timeout)
#     except subprocess.TimeoutExpired:
#         raise RuntimeError("Audio conversion timed out.")
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"ffmpeg failed: {e}")

def ffmpeg_chunk_to_wav16k(
    src: str,
    out_dir: str,
    chunk_seconds: int = 60,
    ffmpeg_bin: str = "ffmpeg",
):
    """
    Split media into 16kHz mono WAV chunks.
    Returns list of chunk paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    pattern = os.path.join(out_dir, "chunk_%03d.wav")

    cmd = [
        ffmpeg_bin, "-y",
        "-i", src,
        "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", "16000",
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        pattern
    ]

    subprocess.run(cmd, check=True)
    return sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.endswith(".wav")
    )



# def normalize_to_wav16k(input_path: str,content_hash: str,upload_dir: str,) -> str:
#     """
#     Cache-normalize media to 16kHz mono WAV.
#     """
    
#     wav_path = normalized_wav_cache_path(content_hash, upload_dir)
#     if os.path.exists(wav_path):
#         return wav_path

#     tmp_path = wav_path + ".tmp"
#     ffmpeg_to_wav16k_mono(input_path, tmp_path)
#     os.replace(tmp_path, wav_path)
#     return wav_path


def transcribe_wav_groq(
    client,
    wav_path: str,
    lang: Optional[str] = None,
    model: str = "whisper-large-v3",
) -> Optional[str]:
    """
    Transcribe a WAV file using Groq Whisper.
    """
    try:
        with open(wav_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=model,
                language=lang,  # None = auto-detect
            )
        return transcription.text.strip()

    except Exception:
        return None
