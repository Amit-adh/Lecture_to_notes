# core/file_utils.py
import os
import uuid
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


