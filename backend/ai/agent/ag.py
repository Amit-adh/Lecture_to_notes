#!/usr/bin/env python3
"""
VIT Exam Question Extraction CLI Tool
Main entry point for scraping and extracting questions from VIT exam papers.
"""
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper import ExamScraper
from extractor import QuestionExtractor
from m import setup_directories, print_success, print_error, print_info, print_progress


def scrape_command(args):
    print_info(f"Starting scrape for subject: {args.subject}, topic: {args.topic}")
    setup_directories()
    scraper = ExamScraper()
    try:
        if getattr(args, 'list_subjects', False):
            subs = scraper.list_available_subjects()
            if subs:
                print_info('Available subjects:')
                for s in subs:
                    print(f'  - {s}')
            else:
                print_info('No subjects found')
            return
        downloaded = scraper.scrape_papers(args.subject, args.topic, max_papers=args.max_papers)
        if downloaded:
            print_success(f"Downloaded {len(downloaded)} papers")
            for d in downloaded:
                print(f"  - {d}")
        else:
            print_info('No papers downloaded')
    except Exception as e:
        print_error(f"Scrape failed: {e}")
        sys.exit(1)


def extract_command(args):
    p = Path(args.file_path)
    if not p.exists():
        print_error(f"File not found: {p}")
        sys.exit(1)
    setup_directories()
    extractor = QuestionExtractor()
    try:
        qs = extractor.extract_questions(str(p), args.topic, use_llm=args.use_llm, use_mathpix=getattr(args, 'use_mathpix', False))
        if qs:
            print_success(f"Extracted {len(qs)} questions")
            for i, q in enumerate(qs,1):
                print(f"{i}. {q}")
            save_path = args.save_path or 'outputs/extracted_questions.json'
            extractor.save_questions(qs, args.topic, args.subject or 'Unknown', save_path)
            print_success(f"Saved to {save_path}")
        else:
            print_info('No questions found')
    except Exception as e:
        print_error(f"Extraction failed: {e}")
        sys.exit(1)


def pipeline_command(args):
    print_info('Running pipeline (scrape -> extract)')
    setup_directories()
    scraper = ExamScraper()
    extractor = QuestionExtractor()
    try:
        downloaded = scraper.scrape_papers(args.subject, args.topic, max_papers=args.max_papers)
        if not downloaded:
            print_info('No papers to process')
            return
        results = extractor.extract_from_multiple_papers(downloaded, args.topic, use_llm=args.use_llm, use_mathpix=getattr(args, 'use_mathpix', False))
        print_success('Pipeline completed')
        print_info(f"Papers processed: {results['papers_processed']}")
        print_info(f"Total questions: {results['total_questions']}")
        out_path = f"outputs/{args.subject.lower().replace(' ','_')}_{args.topic.lower().replace(' ','_')}_pipeline.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print_success(f"Saved pipeline results to {out_path}")
    except Exception as e:
        print_error(f"Pipeline failed: {e}")
        sys.exit(1)


def analyze_command(args):
    p = Path(args.file_path)
    if not p.exists():
        print_error(f"File not found: {p}")
        sys.exit(1)
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print_info(f"Analysis: {p}")
        if 'questions' in data:
            q = data['questions']
            print(f"Total: {len(q)}")
    except Exception as e:
        print_error(f"Analyze failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='VIT Exam Question Extraction CLI')
    sub = parser.add_subparsers(dest='command')
    sp = sub.add_parser('scrape')
    sp.add_argument('--subject', required=True)
    sp.add_argument('--topic', required=False, default=None)
    sp.add_argument('--max_papers', type=int, default=10)
    sp.add_argument('--list_subjects', action='store_true')
    ep = sub.add_parser('extract')
    ep.add_argument('--file_path', required=True)
    ep.add_argument('--topic', required=True)
    ep.add_argument('--subject', help='subject metadata', default='Unknown')
    ep.add_argument('--use_llm', action='store_true')
    ep.add_argument('--save_path')
    ep.add_argument('--use_mathpix', action='store_true', help='Use MathPix OCR fallback (requires MATHPIX_APP_ID & MATHPIX_APP_KEY)')
    pp = sub.add_parser('pipeline')
    pp.add_argument('--subject', required=True)
    pp.add_argument('--topic', required=True)
    pp.add_argument('--use_llm', action='store_true')
    pp.add_argument('--max_papers', type=int, default=5)
    pp.add_argument('--use_mathpix', action='store_true', help='Use MathPix OCR fallback (requires MATHPIX_APP_ID & MATHPIX_APP_KEY)')
    ap = sub.add_parser('analyze')
    ap.add_argument('--file_path', required=True)

    args = parser.parse_args()
    if not args.command:
        parser.print_help(); sys.exit(1)
    if args.command == 'scrape':
        scrape_command(args)
    elif args.command == 'extract':
        extract_command(args)
    elif args.command == 'pipeline':
        pipeline_command(args)
    elif args.command == 'analyze':
        analyze_command(args)


if __name__ == '__main__':
    main()
