# core/file_utils.py
import os
import uuid
import subprocess
import hashlib
from pathlib import Path
from typing import Tuple

def cleanup_upload_dir(upload_dir: str) -> None:
    """
    Remove files in upload_dir. Mirrors original behavior from main.py.
    """
    if not os.path.exists(upload_dir):
        return
    for p in Path(upload_dir).glob("*"):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass


def save_uploaded_file(uploaded_file, upload_dir: str, max_file_mb: int) -> Tuple[str, str]:
    """
    Save a Streamlit UploadedFile-like object to upload_dir with a UUID name.

    Returns: (file_path, safe_name)

    Preserves original errors: raises ValueError for file too large or unsupported type.
    """
    if uploaded_file.size > max_file_mb * 1024 * 1024:
        raise ValueError(f"File too large (>{max_file_mb} MB).")

    ext = Path(uploaded_file.name).suffix.lower()
    allowed_exts = {".mp3", ".wav", ".mp4", ".mov", ".pdf", ".pptx"}
    if ext not in allowed_exts:
        raise ValueError("Unsupported file type.")

    uid = uuid.uuid4().hex
    safe_name = f"{uid}{ext}"
    file_path = os.path.join(upload_dir, safe_name)

    # ensure directory exists
    os.makedirs(upload_dir, exist_ok=True)

    # with open(file_path, "wb") as f:
    #     f.write(uploaded_file.getbuffer())
    
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    os.replace(tmp_path, file_path)  # atomic move


    return file_path, safe_name


def file_hash(path: str) -> str:
    """
    Compute SHA-256 hash of file at path. Reads in 1MB chunks.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def cached_transcript_path(content_hash: str, transcript_cache_dir: str) -> str:
    return os.path.join(transcript_cache_dir, f"{content_hash}.txt")


def normalized_wav_cache_path(content_hash: str, upload_dir: str) -> str:
    return os.path.join(upload_dir, f"{content_hash}_16k.wav")


def ffmpeg_to_wav16k_mono(src: str, dst: str, ffmpeg_bin: str = "ffmpeg", timeout: int = 300):
    """
    Convert audio/video at `src` to 16kHz mono WAV at `dst` using ffmpeg.
    Preserves original exceptions (TimeoutExpired -> RuntimeError, CalledProcessError -> RuntimeError).
    """
    cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src, "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", "16000", "-f", "wav", dst
    ]
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio conversion timed out.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e}")


def normalize_to_wav16k(input_path: str, content_hash: str, upload_dir: str, ffmpeg_bin: str = "ffmpeg") -> str:
    """
    Cache-normalize to 16k mono WAV by content hash. Returns path to WAV file.
    """
    
    wav_path = normalized_wav_cache_path(content_hash, upload_dir)
    if os.path.exists(wav_path):
        return wav_path

    tmp_path = wav_path + ".tmp"
    ffmpeg_to_wav16k_mono(input_path, tmp_path, ffmpeg_bin=ffmpeg_bin)
    os.replace(tmp_path, wav_path)
    return wav_path
