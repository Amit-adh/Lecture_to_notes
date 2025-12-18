# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
TRANSCRIPT_CACHE_DIR = os.getenv("TRANSCRIPT_CACHE_DIR", "transcripts")

MAX_THREADS = int(os.getenv("MAX_THREADS", "4"))
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "100"))

MAX_DOC_CHARS = int(os.environ.get("MAX_DOC_CHARS", "200000"))
INITIAL_SUMMARY_MAX_CHARS = int(os.environ.get("INITIAL_SUMMARY_MAX_CHARS", "16000"))