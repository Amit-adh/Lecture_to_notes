"""
PDF question extractor with robust header/footer removal, page-number stripping, and OCR fallback.
Extraction flow:
- Try pdfplumber word-level extraction grouped by Y to preserve symbols.
- Clean each page: strip very short lines, remove page-number/header patterns, detect repeated first/last lines across pages.
- If pdf text is empty or too sparse, render pages with PyMuPDF at high DPI and OCR via pytesseract.
- Optionally pass cleaned text through an LLM cleaner (safe no-op by default) before question splitting.
"""
import os
import re
import time
import io
import base64
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from m import print_info, print_success, print_error, print_progress, save_json
import requests

# Optional imports with graceful degradation
try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None  # type: ignore

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore
    Image = None  # type: ignore


class QuestionExtractor:
    def __init__(self):
        # LLM related config (no-op unless user provides implementation)
        self.model_name = "koesn/llama3-8b-instruct:latest"
        # Enable LLM hook; it will only run when use_llm=True is passed in public API
        self.llm_enabled = True
        # MathPix credentials from env
        self.mathpix_app_id = os.getenv("MATHPIX_APP_ID")
        self.mathpix_app_key = os.getenv("MATHPIX_APP_KEY")

    # -----------------------
    # PDF Extraction (pdfplumber)
    # -----------------------
    def _words_to_lines(self, words: List[dict], y_tol: float = 2.0) -> List[str]:
        """Group pdfplumber words into lines by their baseline y coordinate.
        Keeps reading order left-to-right; merges small gaps into spaces.
        """
        if not words:
            return []
        # Group by rounded y coordinate
        rows: Dict[float, List[dict]] = defaultdict(list)
        for w in words:
            y = round(float(w.get("top", 0.0)) / y_tol) * y_tol
            rows[y].append(w)
        lines: List[str] = []
        for y in sorted(rows.keys()):
            row = sorted(rows[y], key=lambda w: float(w.get("x0", 0.0)))
            text_parts: List[str] = []
            prev_x1: Optional[float] = None
            for w in row:
                t = w.get("text", "")
                x0 = float(w.get("x0", 0.0))
                if prev_x1 is not None and x0 - prev_x1 > 3.0:
                    text_parts.append(" ")  # gap heuristic
                text_parts.append(t)
                prev_x1 = float(w.get("x1", x0))
            line = "".join(text_parts).strip()
            if line:
                lines.append(line)
        return lines

    def _extract_with_pdfplumber(self, file_path: str) -> List[List[str]]:
        """Return a list of pages, each a list of lines extracted via pdfplumber."""
        if not pdfplumber:
            return []
        pages_lines: List[List[str]] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    try:
                        words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
                        lines = self._words_to_lines(words)
                    except Exception:
                        # fallback to simple text if words fail
                        raw = page.extract_text(x_tolerance=1.5, y_tolerance=3.0) or ""
                        lines = [l.strip() for l in raw.splitlines() if l.strip()]
                    pages_lines.append(lines)
        except Exception as e:
            print_error(f"pdfplumber failed on {os.path.basename(file_path)}: {e}")
            return []
        return pages_lines

    # -----------------------
    # OCR Fallback (PyMuPDF + pytesseract)
    # -----------------------
    def _render_page_to_image(self, doc: "fitz.Document", page_index: int, dpi: int = 300):
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix

    def _extract_with_ocr(self, file_path: str, dpi: int = 300) -> List[List[str]]:
        if not (fitz and pytesseract and Image):
            return []
        pages_lines: List[List[str]] = []
        try:
            doc = fitz.open(file_path)
            for i in range(doc.page_count):
                try:
                    pix = self._render_page_to_image(doc, i, dpi=dpi)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                    lines = [l.strip() for l in text.splitlines() if l.strip()]
                except Exception:
                    lines = []
                pages_lines.append(lines)
            doc.close()
        except Exception as e:
            print_error(f"OCR fallback failed: {e}")
            return []
        return pages_lines

    # -----------------------
    # Cleaning headers/footers
    # -----------------------
    def _detect_repeated_edges(self, pages_lines: List[List[str]]) -> Tuple[Optional[str], Optional[str]]:
        """Detect repeated first and last lines across pages as headers/footers."""
        firsts = [next((l for l in p if l.strip()), "") for p in pages_lines if p]
        lasts = [next((l for l in reversed(p) if l.strip()), "") for p in pages_lines if p]
        n = max(len(firsts), len(lasts)) or 1
        header = None
        footer = None
        if firsts:
            c = Counter(firsts)
            text, count = c.most_common(1)[0]
            if text and count >= max(2, int(0.4 * n)):
                header = text
        if lasts:
            c = Counter(lasts)
            text, count = c.most_common(1)[0]
            if text and count >= max(2, int(0.4 * n)):
                footer = text
        return header, footer

    def _matches_page_artifact(self, line: str) -> bool:
        patterns = [
            r"^page\s*\d+\s*(of\s*\d+)?$",
            r"^\d+\s*/\s*\d+$",
            r"^\(\s*\d+\s*\)$",
            r"^(section|part)\s*[abcdefghijvix]+\b",
            r"^(instructions?|note[s]?):",
            r"^\*+\s*continued\s*\*+$",
            r"^vellore\s+institute|vit\b",
            r"^\d{4}-\d{2}|\bsemester\b|\bexam\b",
        ]
        s = line.strip().lower()
        if len(s) <= 2:
            return True
        for pat in patterns:
            if re.search(pat, s, flags=re.I):
                return True
        return False

    def _clean_pages(self, pages_lines: List[List[str]]) -> List[str]:
        header, footer = self._detect_repeated_edges(pages_lines)
        cleaned_all: List[str] = []
        for lines in pages_lines:
            page_clean: List[str] = []
            for l in lines:
                if len(l.strip()) <= 2:
                    continue  # (a) strip very short lines
                if self._matches_page_artifact(l):
                    continue  # (b) remove page numbers/headers
                page_clean.append(l)
            # (c) remove detected header/footer if present
            if header and page_clean and page_clean[0].strip() == header.strip():
                page_clean = page_clean[1:]
            if footer and page_clean and page_clean[-1].strip() == footer.strip():
                page_clean = page_clean[:-1]
            cleaned_all.extend(page_clean + [""])  # keep page break as empty line
        return cleaned_all

    # -----------------------
    # LLM cleaning hook (optional, safe no-op)
    # -----------------------
    def _llm_clean_text(self, text: str) -> str:
        """Override to integrate an LLM cleaner. Default: return input unchanged."""
        if not self.llm_enabled:
            return text
        try:
            prompt = (
                "You are a document cleaner for exam questions.\n"
                "Input is OCR/extracted text of an exam paper.\n"
                "Tasks:\n"
                "- Remove any remaining headers/footers, page numbers, watermarks.\n"
                "- Fix common OCR artifacts (broken hyphenation, wrong spaces).\n"
                "- Preserve math expressions faithfully (fractions, integrals, matrices).\n"
                "- Keep original question order and line breaks sensibly.\n"
                "Return the cleaned text only.\n\n"
                "Text:\n" + text
            )
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2}
                },
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                cleaned = data.get("response", "")
                return cleaned.strip() or text
            else:
                print_error(f"Ollama cleaning failed: HTTP {resp.status_code}")
                return text
        except Exception as e:
            print_error(f"Ollama cleaning error: {e}")
            return text

    # -----------------------
    # MathPix OCR pass (optional)
    # -----------------------
    def _mathpix_available(self) -> bool:
        return bool(self.mathpix_app_id and self.mathpix_app_key and fitz)

    def _img_to_data_uri(self, pil_img: "Image") -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _extract_with_mathpix(self, file_path: str, dpi: int = 300) -> List[List[str]]:
        """Render each page and send to MathPix v3/text. Returns per-page lines."""
        if not (self._mathpix_available() and Image):
            return []
        pages_lines: List[List[str]] = []
        headers = {
            "app_id": self.mathpix_app_id or "",
            "app_key": self.mathpix_app_key or "",
            "Content-type": "application/json",
        }
        try:
            doc = fitz.open(file_path)
            for i in range(doc.page_count):
                try:
                    pix = self._render_page_to_image(doc, i, dpi=dpi)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    src = self._img_to_data_uri(img)
                    payload = {
                        "src": src,
                        "formats": ["text"],
                        "data_options": {
                            "include_asciimath": False,
                            "include_latex": True
                        },
                        "math_inline_delimiters": ["$", "$"],
                        "rm_spaces": True,
                    }
                    r = requests.post("https://api.mathpix.com/v3/text", headers=headers, data=json.dumps(payload), timeout=120)
                    if r.status_code == 200:
                        data = r.json()
                        text = data.get("text", "")
                        lines = [l.strip() for l in text.splitlines() if l.strip()]
                    else:
                        print_error(f"MathPix page {i+1} failed: HTTP {r.status_code}")
                        lines = []
                except Exception as e:
                    print_error(f"MathPix error on page {i+1}: {e}")
                    lines = []
                pages_lines.append(lines)
            doc.close()
        except Exception as e:
            print_error(f"MathPix pass failed: {e}")
            return []
        return pages_lines

    # -----------------------
    # Question splitting and filtering
    # -----------------------
    def _clean_question_text(self, q: str) -> str:
        q = re.sub(r"\s+", " ", q).strip()
        q = re.sub(r"^\d+[\.)]\s*", "", q)
        q = re.sub(r"^(Q|Question)\s*\d+[:\.)]?\s*", "", q, flags=re.I)
        q = re.sub(r"\[?\(?\s*\d+\s*(marks?|pts?|points?)\s*\)?\]?$", "", q, flags=re.I).strip()
        if q and q[-1] not in ".?!":
            q = q + ("?" if "?" in q else "")
        return q.strip()

    def _extract_vit_questions(self, lines: List[str]) -> List[str]:
        questions: List[str] = []
        buffer = ""
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            # New question starts
            if re.match(r"^\(?\d+\)?[\.)]\s+", line) or re.match(r"^[Qq](uestion)?\s*\d+[:\.)]?\s+", line):
                if buffer:
                    questions.append(buffer.strip())
                buffer = re.sub(r"^\(?\d+\)?[\.)]\s*", "", line)
                buffer = re.sub(r"^[Qq](uestion)?\s*\d+[:\.)]?\s*", "", buffer)
                continue
            # Likely end when question mark at end or bullet options start
            if re.search(r"\?$", line) and buffer:
                buffer += " " + line
                questions.append(buffer.strip())
                buffer = ""
                continue
            # Continuation lines
            if buffer:
                buffer += " " + line
        if buffer:
            questions.append(buffer.strip())
        # Clean and dedupe
        cleaned = [self._clean_question_text(q) for q in questions if len(q) > 20]
        seen = set(); out: List[str] = []
        for q in cleaned:
            if q not in seen:
                seen.add(q); out.append(q)
        return out

    def _filter_by_topic(self, questions: List[str], topic: str) -> List[str]:
        if not topic:
            return questions
        kws = [w.lower() for w in re.split(r"\W+", topic) if w]
        out = []
        for q in questions:
            lower = q.lower()
            if any(k in lower for k in kws):
                out.append(q)
        return out

    # -----------------------
    # Public API
    # -----------------------
    def extract_questions(self, file_path: str, topic: str, use_llm: bool = False, use_mathpix: bool = False) -> List[str]:
        print_progress("Extracting (pdfplumber)...")
        pages_lines = self._extract_with_pdfplumber(file_path)
        # If plumber failed or produced too little content, OCR fallback
        total_chars = sum(len(l) for p in pages_lines for l in p)
        if not pages_lines or total_chars < 50:
            if use_mathpix and self._mathpix_available():
                print_info("Falling back to MathPix OCR (best for math)...")
                pages_lines = self._extract_with_mathpix(file_path)
            else:
                print_info("Falling back to OCR (high-DPI render + Tesseract)...")
                pages_lines = self._extract_with_ocr(file_path)

        if not pages_lines:
            print_error("No text could be extracted from the PDF.")
            return []

        print_progress("Cleaning headers/footers and artifacts...")
        cleaned_lines = self._clean_pages(pages_lines)
        text = "\n".join(cleaned_lines)

        if use_llm and self.llm_enabled:
            print_progress("Running LLM-based cleaning...")
            text = self._llm_clean_text(text)

        print_progress("Parsing questions...")
        questions = self._extract_vit_questions(text.splitlines())
        filtered = self._filter_by_topic(questions, topic)
        print_success(f"Extracted {len(filtered)} questions from {os.path.basename(file_path)}")
        return filtered

    def extract_from_multiple_papers(self, file_paths: List[str], topic: str, use_llm: bool = False, use_mathpix: bool = False) -> dict:
        results = {
            'papers_processed': 0,
            'total_questions': 0,
            'questions_by_paper': {},
            'combined_questions': [],
            'failed_papers': []
        }
        for f in file_paths:
            try:
                qs = self.extract_questions(f, topic, use_llm=use_llm, use_mathpix=use_mathpix)
                results['papers_processed'] += 1
                results['total_questions'] += len(qs)
                results['questions_by_paper'][os.path.basename(f)] = qs
                results['combined_questions'].extend(qs)
            except Exception as e:
                results['failed_papers'].append({'file': f, 'error': str(e)})
        # dedupe
        seen = set(); combined: List[str] = []
        for q in results['combined_questions']:
            if q not in seen:
                seen.add(q); combined.append(q)
        results['combined_questions'] = combined
        results['unique_questions'] = len(combined)
        return results

    def save_questions(self, questions: List[str], topic: str, subject: str, save_path: str):
        data = {
            'metadata': {
                'subject': subject,
                'topic': topic,
                'question_count': len(questions),
                'extracted_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'questions': []
        }
        for q in questions:
            data['questions'].append({
                'text': q,
                'word_count': len(q.split()),
                'question_type': 'open' if '?' in q else 'statement'
            })
        save_json(data, save_path)


if __name__ == '__main__':
    ex = QuestionExtractor()
    test = 'data/raw_papers'
    if os.path.exists(test):
        files = [str(p) for p in Path(test).glob('*.pdf')][:3]
        if files:
            res = ex.extract_from_multiple_papers(files, topic='')
            print('Sample extracted:', res['combined_questions'][:5])
        else:
            print('No PDFs to test')
    else:
        print('No raw_papers dir found')
