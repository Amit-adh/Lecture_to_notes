"""
Utility functions for the VIT Exam Question Extraction CLI application.
Provides printing helpers, file utilities and lightweight dependency checks.
"""
import os
import sys
import json
import time
import re
import shutil
import hashlib
import platform
from pathlib import Path
from typing import Any, Dict, Optional, List

# Color codes for terminal output (works on most terminals)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _colorize(color: str, message: str) -> str:
    try:
        return f"{color}{message}{Colors.END}"
    except Exception:
        return message


def print_success(message: str):
    print(_colorize(Colors.GREEN, f"✓ {message}"))


def print_error(message: str):
    print(_colorize(Colors.RED, f"✗ {message}"))


def print_info(message: str):
    print(_colorize(Colors.BLUE, f"ℹ {message}"))


def print_progress(message: str):
    print(_colorize(Colors.CYAN, f"⋯ {message}"))


def print_warning(message: str):
    print(_colorize(Colors.YELLOW, f"⚠ {message}"))


def setup_directories():
    directories = [
        "data",
        "data/raw_papers",
        "data/cache",
        "outputs",
        "outputs/analysis",
        "logs"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, file_path: str, pretty: bool = True):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)


def load_json(file_path: str) -> Optional[Any]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print_warning(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON {file_path}: {e}")
        return None


def file_exists(file_path: str) -> bool:
    return Path(file_path).exists()


def get_file_size(file_path: str) -> int:
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return 0


def format_file_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024
        i += 1
    if i == 0:
        return f"{int(size)} {size_names[i]}"
    return f"{size:.1f} {size_names[i]}"


def clean_filename(filename: str) -> str:
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    filename = re.sub(r'_+', '_', filename)
    filename = filename.strip('_')
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:195] + ext
    return filename


def is_valid_pdf(file_path: str) -> bool:
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False


def count_pdf_pages(file_path: str) -> int:
    try:
        import PyPDF2
        with open(file_path, 'rb') as f:
            rdr = PyPDF2.PdfReader(f)
            return len(rdr.pages)
    except Exception:
        return 0


def check_dependencies() -> Dict[str, bool]:
    deps = {
        'requests': False,
        'beautifulsoup4': False,
        'PyPDF2': False,
        'playwright': False,
        'langchain': False
    }
    try:
        import requests as _
        deps['requests'] = True
    except Exception:
        deps['requests'] = False
    try:
        import bs4 as _
        deps['beautifulsoup4'] = True
    except Exception:
        deps['beautifulsoup4'] = False
    try:
        import PyPDF2 as _
        deps['PyPDF2'] = True
    except Exception:
        deps['PyPDF2'] = False
    try:
        import playwright as _
        deps['playwright'] = True
    except Exception:
        deps['playwright'] = False
    try:
        import langchain as _
        deps['langchain'] = True
    except Exception:
        deps['langchain'] = False
    return deps


def print_dependency_status():
    deps = check_dependencies()
    print_info("Dependency status:")
    for k, v in deps.items():
        status = 'OK' if v else 'MISSING'
        print(f"  - {k:12}: {status}")


if __name__ == '__main__':
    print_banner = lambda title, subtitle='': print(f"=== {title} {subtitle} ===")
    print_banner('Utility test')
    setup_directories()
    print_dependency_status()
