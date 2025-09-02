#!/usr/bin/env python3
"""
VIT Exam Question Extraction CLI Tool
Main entry point for scraping and extracting questions from VIT exam papers.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from scraper import ExamScraper
from extractor import QuestionExtractor
from m import setup_directories, print_success, print_error, print_info, print_progress


def scrape_command(args):
    """Handle the scrape subcommand."""
    print_info(f"Starting scrape for subject: {args.subject}, topic: {args.topic}")
    
    # Setup directories
    setup_directories()
    
    scraper = ExamScraper()
    try:
        # First, let's list available subjects if requested
        if args.list_subjects:
            subjects = scraper.list_available_subjects()
            if subjects:
                print_info("Available subjects found on CodeChef VIT:")
                for subject in subjects:
                    print(f"  - {subject}")
            else:
                print_info("Could not fetch available subjects")
            return
        
        # Perform the actual scraping
        max_papers = getattr(args, 'max_papers', 10)
        downloaded_files = scraper.scrape_papers(args.subject, args.topic, max_papers)
        
        if downloaded_files:
            print_success(f"Successfully downloaded {len(downloaded_files)} papers:")
            for i, file_path in enumerate(downloaded_files, 1):
                print(f"  {i}. {os.path.basename(file_path)}")
        else:
            print_info("No new papers were downloaded")
            print_info("This could mean:")
            print("    - Papers already exist in data/raw_papers/")
            print("    - No papers found matching your subject/topic")
            print("    - The website structure has changed")
            
    except Exception as e:
        print_error(f"Scraping failed: {str(e)}")
        print_info("Try checking your internet connection or the website status")
        sys.exit(1)


def extract_command(args):
    """Handle the extract subcommand."""
    file_path = Path(args.file_path)
    
    if not file_path.exists():
        print_error(f"File not found: {file_path}")
        print_info("Make sure the file path is correct and the file exists")
        sys.exit(1)
    
    # Check if it's a PDF
    if not str(file_path).lower().endswith('.pdf'):
        print_error("File must be a PDF")
        sys.exit(1)
    
    print_info(f"Extracting questions from: {file_path}")
    print_info(f"Topic filter: {args.topic}")
    print_info(f"Using LLM: {'Yes' if args.use_llm else 'No'}")
    
    # Setup directories
    setup_directories()
    
    extractor = QuestionExtractor()
    try:
        questions = extractor.extract_questions(
            file_path=str(file_path),
            topic=args.topic,
            use_llm=args.use_llm
        )
        
        if questions:
            print_success(f"Extracted {len(questions)} questions related to '{args.topic}'")
            
            # Print questions with numbers
            print_info("Extracted Questions:")
            print("=" * 60)
            for i, question in enumerate(questions, 1):
                print(f"\n{i}. {question}")
            print("\n" + "=" * 60)
            
            # Save to file
            save_path = args.save_path or "outputs/extracted_questions.json"
            extractor.save_questions(questions, args.topic, args.subject or "Unknown", save_path)
            print_success(f"Questions saved to: {save_path}")
            
        else:
            print_info(f"No questions found related to '{args.topic}'")
            print_info("Try:")
            print("    - Using a broader topic (e.g., 'Math' instead of 'Linear Algebra')")
            print("    - Checking if the PDF contains readable text")
            print("    - Using --use_llm flag for better extraction")
            
    except Exception as e:
        print_error(f"Extraction failed: {str(e)}")
        print_info("This could be due to:")
        print("    - Corrupted or scanned PDF file")
        print("    - PDF with images instead of text")
        print("    - Unsupported PDF format")
        sys.exit(1)


def pipeline_command(args):
    """Handle the pipeline subcommand (scrape + extract)."""
    print_info(f"Starting full pipeline for subject: {args.subject}, topic: {args.topic}")
    print_info("This will scrape papers and then extract questions from all of them")
    
    # Setup directories
    setup_directories()
    
    # Step 1: Scrape papers
    print("\n" + "="*60)
    print("STEP 1: SCRAPING PAPERS")
    print("="*60)
    
    scraper = ExamScraper()
    try:
        max_papers = getattr(args, 'max_papers', 10)
        downloaded_files = scraper.scrape_papers(args.subject, args.topic, max_papers)
        
        if not downloaded_files:
            # Check if papers already exist
            data_dir = Path("data/raw_papers")
            existing_files = list(data_dir.glob("*.pdf"))
            if existing_files:
                print_info("No new papers downloaded, but found existing papers:")
                for f in existing_files[:5]:  # Show first 5
                    print(f"    - {f.name}")
                if len(existing_files) > 5:
                    print(f"    ... and {len(existing_files) - 5} more files")
                
                downloaded_files = [str(f) for f in existing_files]
            else:
                print_error("No papers found to process")
                print_info("Try:")
                print("    - Different subject/topic combinations")
                print("    - Check if papers.codechefvit.com is accessible")
                sys.exit(1)
    
    except Exception as e:
        print_error(f"Scraping failed: {str(e)}")
        sys.exit(1)
    
    # Step 2: Extract questions from all papers
    print("\n" + "="*60)
    print("STEP 2: EXTRACTING QUESTIONS")
    print("="*60)
    
    extractor = QuestionExtractor()
    
    try:
        # Process multiple papers at once
        results = extractor.extract_from_multiple_papers(
            file_paths=downloaded_files,
            topic=args.topic,
            use_llm=args.use_llm
        )
        
        # Display results
        print_success(f"Pipeline completed!")
        print_info(f"Papers processed: {results['papers_processed']}")
        print_info(f"Total questions found: {results['total_questions']}")
        print_info(f"Unique questions after deduplication: {results['unique_questions']}")
        
        if results['failed_papers']:
            print_info(f"Failed to process {len(results['failed_papers'])} papers")
        
        if results['combined_questions']:
            # Show sample questions
            print_info("Sample extracted questions:")
            print("-" * 40)
            for i, question in enumerate(results['combined_questions'][:5], 1):
                print(f"{i}. {question}")
            
            if len(results['combined_questions']) > 5:
                print(f"... and {len(results['combined_questions']) - 5} more questions")
            
            # Save all questions with comprehensive metadata
            save_path = f"outputs/{args.subject.lower().replace(' ', '_')}_{args.topic.lower().replace(' ', '_')}_pipeline.json"
            
            # Create detailed output
            pipeline_data = {
                "pipeline_metadata": {
                    "subject": args.subject,
                    "topic": args.topic,
                    "papers_processed": results['papers_processed'],
                    "total_questions": results['total_questions'],
                    "unique_questions": results['unique_questions'],
                    "use_llm": args.use_llm,
                    "failed_papers": results['failed_papers']
                },
                "questions_by_paper": results['questions_by_paper'],
                "combined_questions": results['combined_questions']
            }
            
            # Save detailed results
            with open(save_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(pipeline_data, f, indent=2, ensure_ascii=False)
            
            print_success(f"Complete pipeline results saved to: {save_path}")
            
        else:
            print_info(f"No questions found related to '{args.topic}' in any papers")
            
    except Exception as e:
        print_error(f"Question extraction failed: {str(e)}")
        sys.exit(1)


def analyze_command(args):
    """Handle the analyze subcommand for analyzing existing question files."""
    file_path = Path(args.file_path)
    
    if not file_path.exists():
        print_error(f"File not found: {file_path}")
        sys.exit(1)
    
    try:
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print_info(f"Analysis of: {file_path}")
        print("=" * 50)
        
        # Basic stats
        if 'questions' in data:
            questions = data['questions']
            print(f"Total questions: {len(questions)}")
            
            if questions:
                # Question type analysis
                if 'question_type' in questions[0]:
                    type_counts = {}
                    for q in questions:
                        qtype = q.get('question_type', 'unknown')
                        type_counts[qtype] = type_counts.get(qtype, 0) + 1
                    
                    print("\nQuestion types:")
                    for qtype, count in sorted(type_counts.items()):
                        print(f"  {qtype}: {count}")
                
                # Difficulty analysis
                if 'estimated_difficulty' in questions[0]:
                    diff_counts = {}
                    for q in questions:
                        difficulty = q.get('estimated_difficulty', 'unknown')
                        diff_counts[difficulty] = diff_counts.get(difficulty, 0) + 1
                    
                    print("\nDifficulty distribution:")
                    for diff, count in sorted(diff_counts.items()):
                        print(f"  {diff}: {count}")
                
                # Word count stats
                if 'word_count' in questions[0]:
                    word_counts = [q.get('word_count', 0) for q in questions]
                    print(f"\nWord count stats:")
                    print(f"  Average: {sum(word_counts)/len(word_counts):.1f}")
                    print(f"  Min: {min(word_counts)}")
                    print(f"  Max: {max(word_counts)}")
        
        # Metadata
        if 'metadata' in data:
            metadata = data['metadata']
            print(f"\nMetadata:")
            print(f"  Subject: {metadata.get('subject', 'N/A')}")
            print(f"  Topic: {metadata.get('topic', 'N/A')}")
            print(f"  Extracted at: {metadata.get('extracted_at', 'N/A')}")
            print(f"  Method: {metadata.get('extraction_method', 'N/A')}")
        
    except Exception as e:
        print_error(f"Analysis failed: {str(e)}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VIT Exam Question Extraction CLI - Extract questions from VIT exam papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available subjects
  %(prog)s scrape --subject "Any" --topic "Any" --list_subjects
  
  # Scrape papers for a specific subject
  %(prog)s scrape --subject "Mathematics" --topic "Linear Algebra" --max_papers 5
  
  # Extract questions from a specific PDF
  %(prog)s extract --file_path "data/raw_papers/math_paper.pdf" --topic "Algebra" --subject "Mathematics"
  
  # Extract with LLM enhancement
  %(prog)s extract --file_path "paper.pdf" --topic "Calculus" --use_llm --save_path "calculus_questions.json"
  
  # Run full pipeline (scrape + extract)
  %(prog)s pipeline --subject "Physics" --topic "Mechanics" --use_llm --max_papers 3
  
  # Analyze existing question file
  %(prog)s analyze --file_path "outputs/questions.json"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scrape subcommand
    scrape_parser = subparsers.add_parser('scrape', help='Scrape VIT exam papers from CodeChef VIT')
    scrape_parser.add_argument('--subject', required=True, help='Subject area (e.g., "Mathematics", "Physics")')
    scrape_parser.add_argument('--topic', required=True, help='Specific topic (e.g., "Linear Algebra", "Mechanics")')
    scrape_parser.add_argument('--max_papers', type=int, default=10, help='Maximum papers to download (default: 10)')
    scrape_parser.add_argument('--list_subjects', action='store_true', help='List available subjects from the website')
    
    # Extract subcommand
    extract_parser = subparsers.add_parser('extract', help='Extract questions from a PDF file')
    extract_parser.add_argument('--file_path', required=True, help='Path to the PDF file')
    extract_parser.add_argument('--topic', required=True, help='Topic to filter questions')
    extract_parser.add_argument('--subject', help='Subject area for metadata (optional)')
    extract_parser.add_argument('--use_llm', action='store_true', help='Use LLM (Ollama) for better extraction')
    extract_parser.add_argument('--save_path', help='Output file path (default: outputs/extracted_questions.json)')
    
    # Pipeline subcommand
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline (scrape + extract)')
    pipeline_parser.add_argument('--subject', required=True, help='Subject area (e.g., "Mathematics", "Physics")')
    pipeline_parser.add_argument('--topic', required=True, help='Specific topic (e.g., "Linear Algebra")')
    pipeline_parser.add_argument('--use_llm', action='store_true', help='Use LLM (Ollama) for better extraction')
    pipeline_parser.add_argument('--max_papers', type=int, default=10, help='Maximum papers to process (default: 10)')
    
    # Analyze subcommand
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing question JSON file')
    analyze_parser.add_argument('--file_path', required=True, help='Path to the JSON file to analyze')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Display banner
    print("=" * 60)
    print("VIT EXAM QUESTION EXTRACTION TOOL")
    print("Source: https://papers.codechefvit.com/")
    print("=" * 60)
    
    # Route to appropriate command handler
    if args.command == 'scrape':
        scrape_command(args)
    elif args.command == 'extract':
        extract_command(args)
    elif args.command == 'pipeline':
        pipeline_command(args)
    elif args.command == 'analyze':
        analyze_command(args)


if __name__ == "__main__":
    main()