#!/usr/bin/env python3
"""
Utility script: extract questions from all PDFs in data/raw_papers and save results.
"""
import os
import json
from pathlib import Path
from extractor import QuestionExtractor
from m import setup_directories, print_info, print_success, print_error

if __name__ == '__main__':
    setup_directories()
    data_dir = Path('data/raw_papers')
    if not data_dir.exists():
        print_error('No data/raw_papers directory found')
        raise SystemExit(1)
    files = list(data_dir.glob('*.pdf'))
    if not files:
        print_error('No PDF files found in data/raw_papers')
        raise SystemExit(1)
    print_info(f'Found {len(files)} PDF(s) to process')
    extractor = QuestionExtractor()
    results = extractor.extract_from_multiple_papers([str(p) for p in files], topic='')
    out_path = Path('outputs') / 'extracted_questions_all.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print_success(f'Saved extraction results to {out_path}')
