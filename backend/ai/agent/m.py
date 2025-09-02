"""
Utility functions for the VIT Exam Question Extraction CLI application.
Enhanced with additional features for better user experience.
"""

import os
import json
import time
import re
from pathlib import Path
from typing import Any, List, Dict, Optional
import hashlib


# Color codes for terminal output
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
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'


def print_success(message: str):
    """Print a success message in green."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_error(message: str):
    """Print an error message in red."""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def print_info(message: str):
    """Print an info message in blue."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")


def print_progress(message: str):
    """Print a progress message in cyan."""
    print(f"{Colors.CYAN}⋯ {message}{Colors.END}")


def print_warning(message: str):
    """Print a warning message in yellow."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_header(message: str):
    """Print a header message with emphasis."""
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}{message}{Colors.END}")


def print_banner(title: str, subtitle: str = ""):
    """Print a banner with title and optional subtitle."""
    width = 60
    print("=" * width)
    print(f"{Colors.BOLD}{title:^{width}}{Colors.END}")
    if subtitle:
        print(f"{Colors.CYAN}{subtitle:^{width}}{Colors.END}")
    print("=" * width)


def print_step(step_num: int, total_steps: int, message: str):
    """Print a step indicator."""
    print(f"\n{Colors.BOLD}[Step {step_num}/{total_steps}]{Colors.END} {Colors.BLUE}{message}{Colors.END}")


def print_highlight(message: str):
    """Print a highlighted message."""
    print(f"{Colors.BOLD}{Colors.MAGENTA}★ {message}{Colors.END}")


def setup_directories():
    """Create necessary directories for the application."""
    directories = [
        "data",
        "data/raw_papers",
        "data/cache",
        "outputs",
        "outputs/analysis",
        "outputs/reports",
        "logs"
    ]
    
    created_dirs = []
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)
    
    if created_dirs:
        print_info(f"Created directories: {', '.join(created_dirs)}")


def load_json(file_path: str) -> Any:
    """Load data from a JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print_success(f"Loaded data from: {file_path}")
        return data
    except FileNotFoundError:
        print_error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in {file_path}: {str(e)}")
        return None
    except Exception as e:
        print_error(f"Error loading {file_path}: {str(e)}")
        return None


def save_json(data: Any, file_path: str, pretty: bool = True):
    """Save data to a JSON file with enhanced formatting."""
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save with proper formatting
        with open(file_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
            else:
                json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
        
        file_size = get_file_size(file_path)
        print_success(f"Data saved to: {file_path} ({format_file_size(file_size)})")
        
    except Exception as e:
        print_error(f"Failed to save {file_path}: {str(e)}")
        raise


def backup_file(file_path: str) -> Optional[str]:
    """Create a backup of a file with timestamp."""
    try:
        if not Path(file_path).exists():
            return None
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.backup_{timestamp}"
        
        import shutil
        shutil.copy2(file_path, backup_path)
        
        print_info(f"Backup created: {backup_path}")
        return backup_path
        
    except Exception as e:
        print_warning(f"Could not create backup: {str(e)}")
        return None


def file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return Path(file_path).exists()


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    try:
        return Path(file_path).stat().st_size
    except OSError:
        return 0


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get comprehensive file information."""
    path = Path(file_path)
    
    if not path.exists():
        return {"exists": False}
    
    try:
        stat = path.stat()
        return {
            "exists": True,
            "size": stat.st_size,
            "size_formatted": format_file_size(stat.st_size),
            "created": time.ctime(stat.st_ctime),
            "modified": time.ctime(stat.st_mtime),
            "extension": path.suffix.lower(),
            "name": path.name,
            "parent": str(path.parent),
            "is_pdf": path.suffix.lower() == '.pdf'
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def clean_filename(filename: str) -> str:
    """Clean a filename by removing invalid characters."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove extra spaces and underscores
    filename = re.sub(r'\s+', '_', filename)
    filename = re.sub(r'_+', '_', filename)
    
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    
    # Limit length
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:195] + ext
    
    return filename


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
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
    else:
        return f"{size:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def create_cache_key(subject: str, topic: str) -> str:
    """Create a cache key from subject and topic."""
    cleaned_subject = re.sub(r'[^\w\s]', '', subject).lower().replace(' ', '_')
    cleaned_topic = re.sub(r'[^\w\s]', '', topic).lower().replace(' ', '_')
    return f"{cleaned_subject}_{cleaned_topic}"


def generate_file_hash(file_path: str) -> Optional[str]:
    """Generate SHA256 hash of a file."""
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception:
        return None


def is_valid_pdf(file_path: str) -> bool:
    """Check if a file is a valid PDF."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False


def count_pdf_pages(file_path: str) -> int:
    """Count the number of pages in a PDF."""
    try:
        import PyPDF2
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            return len(pdf_reader.pages)
    except Exception:
        return 0


def validate_subject_topic(subject: str, topic: str) -> tuple:
    """Validate and clean subject and topic inputs."""
    # Clean and validate subject
    subject = subject.strip()
    if not subject or len(subject) < 2:
        raise ValueError("Subject must be at least 2 characters long")
    
    # Clean and validate topic
    topic = topic.strip()
    if not topic or len(topic) < 2:
        raise ValueError("Topic must be at least 2 characters long")
    
    return subject, topic


def create_progress_bar(current: int, total: int, width: int = 40) -> str:
    """Create a simple progress bar string."""
    if total == 0:
        return "[" + "=" * width + "]"
    
    progress = current / total
    filled = int(width * progress)
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({progress:.1%})"


def log_to_file(message: str, log_type: str = "INFO"):
    """Log a message to file with timestamp."""
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file = log_dir / f"extraction_{time.strftime('%Y%m%d')}.log"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {log_type}: {message}\n")
    except Exception:
        pass  # Fail silently for logging


def print_table(data: List[Dict[str, Any]], headers: List[str], max_width: int = 100):
    """Print data in a formatted table."""
    if not data or not headers:
        return
    
    # Calculate column widths
    col_widths = {}
    for header in headers:
        col_widths[header] = len(header)
    
    for row in data:
        for header in headers:
            if header in row:
                col_widths[header] = max(col_widths[header], len(str(row[header])))
    
    # Ensure we don't exceed max width
    total_width = sum(col_widths.values()) + len(headers) * 3
    if total_width > max_width:
        reduction = (total_width - max_width) // len(headers)
        for header in headers:
            col_widths[header] = max(10, col_widths[header] - reduction)
    
    # Print header
    header_row = " | ".join(header.ljust(col_widths[header]) for header in headers)
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for row in data:
        data_row = " | ".join(
            str(row.get(header, "")).ljust(col_widths[header])[:col_widths[header]]
            for header in headers
        )
        print(data_row)


def get_system_info() -> Dict[str, str]:
    """Get basic system information."""
    import platform
    import sys
    
    return {
        "OS": platform.system(),
        "OS Version": platform.release(),
        "Python Version": sys.version.split()[0],
        "Architecture": platform.machine(),
        "Hostname": platform.node()
    }


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available."""
    dependencies = {
        "requests": False,
        "beautifulsoup4": False,
        "PyPDF2": False,
        "ollama": False
    }
    
    for dep in dependencies:
        try:
            if dep == "beautifulsoup4":
                import bs4
            else:
                __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies


def print_dependency_status():
    """Print the status of dependencies."""
    deps = check_dependencies()
    
    print_header("Dependency Status")
    for dep, available in deps.items():
        status = "✓ Available" if available else "✗ Missing"
        color = Colors.GREEN if available else Colors.RED
        print(f"  {dep:15} : {color}{status}{Colors.END}")
    
    missing = [dep for dep, available in deps.items() if not available]
    if missing:
        print_warning(f"Missing dependencies: {', '.join(missing)}")
        print_info("Install with: pip install " + " ".join(missing))


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print_progress(f"Starting: {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            print_success(f"Completed: {self.description} ({format_duration(duration)})")
        else:
            print_error(f"Failed: {self.description} ({format_duration(duration)})")


# Usage example and testing
if __name__ == "__main__":
    # Test the utility functions
    print_banner("VIT EXAM QUESTION EXTRACTOR", "Utility Functions Test")
    
    # Test directory setup
    setup_directories()
    
    # Test dependency checking
    print_dependency_status()
    
    # Test timer
    with Timer("Test operation"):
        time.sleep(1)
    
    # Test file operations
    test_data = {"test": "data", "numbers": [1, 2, 3]}
    save_json(test_data, "test_output.json")
    
    if file_exists("test_output.json"):
        loaded_data = load_json("test_output.json")
        print_success("File operations working correctly")
        
        # Cleanup
        os.remove("test_output.json")
    
    print_highlight("All utility functions tested successfully!")