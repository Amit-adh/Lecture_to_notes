# core/media_utils.py

import os
import subprocess
from typing import Optional


def ffmpeg_to_wav16k_mono(src: str, dst: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-vn", "-sn", "-dn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        dst,
    ]
    subprocess.run(cmd, check=True, timeout=300)


def normalize_to_wav16k(src: str, dst: str) -> str:
    """
    Normalize any media file to 16kHz mono WAV.
    """
    if os.path.exists(dst):
        return dst

    ffmpeg_to_wav16k_mono(src, dst)
    return dst


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
