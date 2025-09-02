"""
PDF question extractor optimized for VIT exam papers with optional LLM enhancement.
"""

import PyPDF2
import re
import json
from pathlib import Path
from typing import List, Optional
import ollama
import os
import time
from m import print_info, print_success, print_error, print_progress


class QuestionExtractor:
    """Extracts questions from VIT exam paper PDFs with topic filtering."""
    
    def __init__(self):
        self.ollama_client = None
        self.model_name = "koesn/llama3-8b-instruct:latest"
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup Ollama client and check if model is available."""
        try:
            # Test if Ollama is running and model is available
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.model_name in available_models:
                self.ollama_client = ollama
                print_info(f"Ollama model '{self.model_name}' is available")
            else:
                print_info(f"Ollama model '{self.model_name}' not found. Attempting to pull...")
                ollama.pull(self.model_name)
                self.ollama_client = ollama
                print_success(f"Successfully pulled model '{self.model_name}'")
                
        except Exception as e:
            print_error(f"Ollama not available: {str(e)}")
            print_info("Make sure Ollama is installed and running: https://ollama.ai")
            print_info("LLM features will be disabled.")
    
    def extract_questions(self, file_path: str, topic: str, use_llm: bool = False) -> List[str]:
        """
        Extract questions from a VIT exam paper PDF filtered by topic.
        
        Args:
            file_path (str): Path to the PDF file
            topic (str): Topic to filter questions
            use_llm (bool): Whether to use LLM for better extraction
            
        Returns:
            List[str]: List of extracted questions
        """
        print_progress("Reading PDF content...")
        
        # Extract text from PDF
        text_content = self._extract_pdf_text(file_path)
        
        if not text_content:
            print_error("Failed to extract text from PDF")
            return []
        
        print_progress("Identifying questions...")
        
        # Extract questions using VIT-specific patterns
        questions = self._extract_vit_questions(text_content)
        
        # Filter by topic
        filtered_questions = self._filter_by_topic(questions, topic)
        
        # Clean and deduplicate questions
        cleaned_questions = self._clean_and_deduplicate(filtered_questions)
        
        # Enhance with LLM if requested and available
        if use_llm and self.ollama_client and cleaned_questions:
            print_progress("Enhancing questions with LLM...")
            cleaned_questions = self._enhance_with_llm(cleaned_questions, topic)
        
        return cleaned_questions
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text content from PDF file with better error handling."""
        text_content = ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                print_info(f"PDF has {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                        else:
                            print_info(f"No text found on page {page_num + 1}")
                    except Exception as e:
                        print_error(f"Error reading page {page_num + 1}: {str(e)}")
                        continue
                
                return text_content
                
        except Exception as e:
            print_error(f"Error reading PDF {file_path}: {str(e)}")
            return ""
    
    def _extract_vit_questions(self, text: str) -> List[str]:
        """Extract questions using VIT exam paper specific patterns."""
        questions = []
        lines = text.split('\n')
        
        # VIT exam paper patterns
        patterns = [
            # Numbered questions: 1. Question text
            r'^\s*(\d+)\.?\s+(.+)',
            
            # Question with sub-parts: 1(a), 1(b), etc.
            r'^\s*(\d+)\s*\([a-z]\)\.?\s*(.+)',
            
            # Roman numerals: i), ii), iii)
            r'^\s*([ivx]+)\)\.?\s*(.+)',
            
            # Alphabetic: a), b), c)
            r'^\s*([a-z])\)\.?\s*(.+)',
            
            # Questions starting with "Q" or "Question"
            r'^\s*(Q|Question)\s*(\d+)\.?\s*(.+)',
            
            # Part-wise questions: Part A, Part B, etc.
            r'^\s*Part\s+[A-Z]\.?\s*(.+)',
            
            # Section-wise questions
            r'^\s*Section\s+[A-Z]\.?\s*(.+)',
            
            # Direct questions (ending with ?)
            r'^(.+\?)\s*$',
            
            # Fill in the blanks pattern
            r'^(.+(?:fill|complete|blank).+)',
            
            # True/False questions
            r'^(.+(?:true|false).+)',
            
            # Multiple choice indicators
            r'^(.+(?:choose|select|correct).+)',
        ]
        
        current_question = ""
        question_number = 0
        
        for line in lines:
            line = line.strip()
            
            if not line or len(line) < 5:
                # If we have a current question, add it
                if current_question:
                    questions.append(current_question)
                    current_question = ""
                continue
            
            # Check if line matches any pattern
            is_question_start = False
            
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # If we have a current question, save it first
                    if current_question:
                        questions.append(current_question)
                    
                    # Start new question
                    if len(match.groups()) >= 2:
                        current_question = match.groups()[-1].strip()
                    else:
                        current_question = match.group(1).strip() if match.groups() else line
                    
                    is_question_start = True
                    question_number += 1
                    break
            
            if not is_question_start and current_question:
                # This might be a continuation of the previous question
                if len(line) > 10 and not line.lower().startswith(('answer', 'solution', 'note')):
                    current_question += " " + line
            
            # Check for question ending indicators
            if current_question and any(indicator in line.lower() for indicator in ['marks', 'points', 'pts', '[', ']']):
                questions.append(current_question)
                current_question = ""
        
        # Add the last question if exists
        if current_question:
            questions.append(current_question)
        
        # Post-process questions
        processed_questions = []
        for q in questions:
            cleaned = self._clean_question_text(q)
            if cleaned and len(cleaned) > 20:  # Ensure substantial content
                processed_questions.append(cleaned)
        
        return processed_questions
    
    def _clean_question_text(self, question: str) -> str:
        """Clean and format question text specifically for VIT papers."""
        # Remove extra whitespace
        question = ' '.join(question.split())
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = [
            r'^\d+[\.\)]\s*',           # Number prefixes: 1. or 1)
            r'^[a-z]\)\s*',             # Letter prefixes: a)
            r'^[ivx]+\)\s*',            # Roman numeral prefixes: i)
            r'^(Q|Question)\s*\d*\.?\s*',  # Question prefixes
            r'^Part\s+[A-Z]\.?\s*',     # Part prefixes
            r'^Section\s+[A-Z]\.?\s*',  # Section prefixes
        ]
        
        for prefix in prefixes_to_remove:
            question = re.sub(prefix, '', question, flags=re.IGNORECASE)
        
        # Remove marks indicators at the end
        marks_patterns = [
            r'\s*\[\s*\d+\s*marks?\s*\]\s*$',
            r'\s*\(\s*\d+\s*marks?\s*\)\s*$',
            r'\s*\d+\s*marks?\s*$',
            r'\s*\d+\s*pts?\s*$',
            r'\s*\d+\s*points?\s*$',
        ]
        
        for pattern in marks_patterns:
            question = re.sub(pattern, '', question, flags=re.IGNORECASE)
        
        # Clean up question formatting
        question = question.strip()
        
        # Ensure question ends with proper punctuation
        if question and not question[-1] in '.?!':
            if 'explain' in question.lower() or 'describe' in question.lower() or 'define' in question.lower():
                question += '.'
            elif any(word in question.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which', 'who']):
                question += '?'
            else:
                question += '.'
        
        return question
    
    def _filter_by_topic(self, questions: List[str], topic: str) -> List[str]:
        """Filter questions by relevance to the specified topic with better matching."""
        if not topic:
            return questions
        
        topic_words = topic.lower().split()
        filtered = []
        
        # Create expanded keyword list for common subjects
        expanded_keywords = self._expand_topic_keywords(topic)
        
        for question in questions:
            question_lower = question.lower()
            
            # Direct keyword matching
            if any(word in question_lower for word in topic_words):
                filtered.append(question)
                continue
            
            # Expanded keyword matching
            if any(keyword in question_lower for keyword in expanded_keywords):
                filtered.append(question)
                continue
            
            # Subject-specific pattern matching
            if self._subject_specific_match(question_lower, topic.lower()):
                filtered.append(question)
        
        return filtered
    
    def _expand_topic_keywords(self, topic: str) -> List[str]:
        """Expand topic with related keywords for better matching."""
        topic_lower = topic.lower()
        expanded = [topic_lower]
        
        # Mathematics topics
        if 'algebra' in topic_lower:
            expanded.extend(['matrix', 'vector', 'equation', 'polynomial', 'linear'])
        elif 'calculus' in topic_lower:
            expanded.extend(['derivative', 'integral', 'limit', 'differentiation', 'integration'])
        elif 'statistics' in topic_lower:
            expanded.extend(['probability', 'mean', 'median', 'variance', 'distribution'])
        elif 'geometry' in topic_lower:
            expanded.extend(['triangle', 'circle', 'angle', 'area', 'volume'])
        
        # Physics topics
        elif 'mechanics' in topic_lower:
            expanded.extend(['force', 'motion', 'velocity', 'acceleration', 'momentum'])
        elif 'thermodynamics' in topic_lower:
            expanded.extend(['heat', 'temperature', 'entropy', 'energy', 'gas'])
        elif 'optics' in topic_lower:
            expanded.extend(['light', 'lens', 'mirror', 'refraction', 'reflection'])
        
        # Computer Science topics
        elif 'algorithm' in topic_lower:
            expanded.extend(['sorting', 'searching', 'complexity', 'data structure', 'time'])
        elif 'programming' in topic_lower:
            expanded.extend(['code', 'function', 'variable', 'loop', 'array'])
        elif 'database' in topic_lower:
            expanded.extend(['sql', 'query', 'table', 'relation', 'index'])
        
        # Engineering topics
        elif 'circuit' in topic_lower:
            expanded.extend(['voltage', 'current', 'resistance', 'capacitor', 'inductor'])
        elif 'signal' in topic_lower:
            expanded.extend(['frequency', 'amplitude', 'filter', 'fourier', 'sampling'])
        
        return expanded
    
    def _subject_specific_match(self, question: str, topic: str) -> bool:
        """Check for subject-specific patterns in questions."""
        # Mathematical expressions
        if any(math_topic in topic for math_topic in ['algebra', 'calculus', 'geometry']):
            if re.search(r'[x-z]\s*[=+\-*/]\s*[0-9x-z]', question):
                return True
            if any(symbol in question for symbol in ['∫', '∑', '∂', '√', 'sin', 'cos', 'tan']):
                return True
        
        # Physics units and constants
        if 'physics' in topic:
            units = ['kg', 'm/s', 'newton', 'joule', 'watt', 'volt', 'amp']
            if any(unit in question for unit in units):
                return True
        
        return False
    
    def _clean_and_deduplicate(self, questions: List[str]) -> List[str]:
        """Clean and remove duplicate questions."""
        seen = set()
        cleaned = []
        
        for question in questions:
            # Normalize for comparison
            normalized = re.sub(r'[^\w\s]', '', question.lower().strip())
            normalized = ' '.join(normalized.split())
            
            # Skip if too short or already seen
            if len(normalized) < 10 or normalized in seen:
                continue
            
            seen.add(normalized)
            cleaned.append(question)
        
        return cleaned
    
    def _enhance_with_llm(self, questions: List[str], topic: str) -> List[str]:
        """Use Ollama LLM to enhance and refine questions."""
        if not self.ollama_client:
            print_info("LLM not available, skipping enhancement")
            return questions
        
        try:
            enhanced_questions = []
            
            for i, question in enumerate(questions):
                print_progress(f"Processing question {i+1}/{len(questions)} with LLM...")
                
                prompt = f"""
You are analyzing an exam question from a VIT university paper. 

Topic Focus: "{topic}"
Question: "{question}"

Tasks:
1. Determine if this question is truly related to "{topic}" (Yes/No)
2. If Yes, clean and improve the question formatting
3. If No, respond with exactly "UNRELATED"

Guidelines:
- Keep technical terms and mathematical expressions intact
- Fix grammar and formatting issues
- Make the question clear and well-structured
- Preserve the original meaning and difficulty level

Response format:
- If related: Return the cleaned question only
- If unrelated: Return exactly "UNRELATED"
"""
                
                try:
                    response = self.ollama_client.chat(
                        model=self.model_name,
                        messages=[{'role': 'user', 'content': prompt}],
                        options={
                            'temperature': 0.1,
                            'top_p': 0.9,
                            'num_predict': 200
                        }
                    )
                    
                    result = response['message']['content'].strip()
                    
                    if result != "UNRELATED" and len(result) > 15:
                        enhanced_questions.append(result)
                    elif result == "UNRELATED":
                        print_info(f"LLM filtered out unrelated question: {question[:50]}...")
                    else:
                        # Fallback to original if LLM response is too short
                        enhanced_questions.append(question)
                    
                    time.sleep(0.3)  # Rate limiting
                    
                except Exception as e:
                    print_error(f"Error processing question {i+1} with LLM: {str(e)}")
                    enhanced_questions.append(question)  # Fallback to original
            
            print_success(f"LLM processed {len(questions)} questions, kept {len(enhanced_questions)}")
            return enhanced_questions
            
        except Exception as e:
            print_error(f"LLM enhancement failed: {str(e)}")
            return questions
    
    def save_questions(self, questions: List[str], topic: str, subject: str, save_path: str):
        """Save questions to a JSON file with comprehensive metadata."""
        output_dir = Path(save_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive data structure
        data = {
            "metadata": {
                "subject": subject,
                "topic": topic,
                "question_count": len(questions),
                "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "extractor_version": "2.0",
                "source": "VIT Exam Papers",
                "extraction_method": "pattern_matching + llm" if self.ollama_client else "pattern_matching"
            },
            "questions": []
        }
        
        # Add questions with metadata
        for i, question in enumerate(questions):
            question_data = {
                "id": i + 1,
                "text": question,
                "topic": topic,
                "subject": subject,
                "word_count": len(question.split()),
                "estimated_difficulty": self._estimate_difficulty(question),
                "question_type": self._classify_question_type(question)
            }
            data["questions"].append(question_data)
        
        # Add statistics
        data["statistics"] = {
            "avg_word_count": sum(q["word_count"] for q in data["questions"]) / len(questions) if questions else 0,
            "question_types": self._get_question_type_distribution(data["questions"]),
            "difficulty_distribution": self._get_difficulty_distribution(data["questions"])
        }
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print_success(f"Questions saved with metadata to: {save_path}")
        except Exception as e:
            print_error(f"Failed to save questions: {str(e)}")
            raise
    
    def _estimate_difficulty(self, question: str) -> str:
        """Estimate question difficulty based on content analysis."""
        question_lower = question.lower()
        
        # High difficulty indicators
        high_indicators = ['prove', 'derive', 'analyze', 'evaluate', 'justify', 'compare', 'critique']
        medium_indicators = ['explain', 'describe', 'calculate', 'solve', 'determine', 'find']
        low_indicators = ['define', 'list', 'state', 'identify', 'name', 'what is']
        
        if any(indicator in question_lower for indicator in high_indicators):
            return "High"
        elif any(indicator in question_lower for indicator in medium_indicators):
            return "Medium"
        elif any(indicator in question_lower for indicator in low_indicators):
            return "Low"
        else:
            # Default based on length and complexity
            if len(question.split()) > 20 or len(re.findall(r'[(){}[\]]', question)) > 2:
                return "Medium"
            else:
                return "Low"
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question."""
        question_lower = question.lower()
        
        if question.endswith('?'):
            if any(word in question_lower for word in ['what', 'who', 'when', 'where', 'which']):
                return "factual"
            elif any(word in question_lower for word in ['how', 'why']):
                return "analytical"
            else:
                return "general_question"
        
        elif any(word in question_lower for word in ['calculate', 'solve', 'find', 'determine']):
            return "computational"
        
        elif any(word in question_lower for word in ['explain', 'describe', 'discuss']):
            return "descriptive"
        
        elif any(word in question_lower for word in ['prove', 'derive', 'show']):
            return "proof"
        
        elif any(word in question_lower for word in ['compare', 'contrast', 'evaluate']):
            return "comparative"
        
        elif 'true' in question_lower and 'false' in question_lower:
            return "true_false"
        
        elif re.search(r'\b[a-d]\)|choice', question_lower):
            return "multiple_choice"
        
        else:
            return "other"
    
    def _get_question_type_distribution(self, questions: List[dict]) -> dict:
        """Get distribution of question types."""
        type_counts = {}
        for q in questions:
            qtype = q["question_type"]
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        return type_counts
    
    def _get_difficulty_distribution(self, questions: List[dict]) -> dict:
        """Get distribution of difficulty levels."""
        diff_counts = {}
        for q in questions:
            difficulty = q["estimated_difficulty"]
            diff_counts[difficulty] = diff_counts.get(difficulty, 0) + 1
        return diff_counts
    
    def extract_from_multiple_papers(self, file_paths: List[str], topic: str, use_llm: bool = False) -> dict:
        """Extract questions from multiple papers and combine results."""
        all_results = {
            "papers_processed": 0,
            "total_questions": 0,
            "questions_by_paper": {},
            "combined_questions": [],
            "failed_papers": []
        }
        
        for file_path in file_paths:
            print_info(f"Processing: {os.path.basename(file_path)}")
            
            try:
                questions = self.extract_questions(file_path, topic, use_llm)
                
                if questions:
                    all_results["papers_processed"] += 1
                    all_results["total_questions"] += len(questions)
                    all_results["questions_by_paper"][os.path.basename(file_path)] = questions
                    all_results["combined_questions"].extend(questions)
                else:
                    all_results["failed_papers"].append(file_path)
                
            except Exception as e:
                print_error(f"Failed to process {file_path}: {str(e)}")
                all_results["failed_papers"].append(file_path)
        
        # Deduplicate combined questions
        all_results["combined_questions"] = self._clean_and_deduplicate(all_results["combined_questions"])
        all_results["unique_questions"] = len(all_results["combined_questions"])
        
        return all_results


if __name__ == "__main__":
    # Test the extractor
    extractor = QuestionExtractor()
    
    test_file = "data/raw_papers/test_paper.pdf"
    if os.path.exists(test_file):
        questions = extractor.extract_questions(
            file_path=test_file,
            topic="Linear Algebra",
            use_llm=False
        )
        
        print(f"\nExtracted {len(questions)} questions:")
        for i, question in enumerate(questions[:5], 1):  # Show first 5
            print(f"{i}. {question}")
        
        if len(questions) > 5:
            print(f"... and {len(questions) - 5} more questions")
            
        # Save results
        extractor.save_questions(questions, "Linear Algebra", "Mathematics", "outputs/test_questions.json")
    else:
        print(f"Test file not found: {test_file}")
        print("Place a PDF file at the above path to test extraction.")