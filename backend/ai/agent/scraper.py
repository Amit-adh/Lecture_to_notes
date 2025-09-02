"""
Web scraper for CodeChef VIT exam papers based on subject and topic.
Updated to work with the actual papers.codechefvit.com website.
"""

import requests
from bs4 import BeautifulSoup
import os
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote
import time
import re
from m import print_info, print_success, print_error, print_progress


class ExamScraper:
    """Handles scraping of exam papers from CodeChef VIT papers site."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.base_url = "https://papers.codechefvit.com"
        self.download_dir = Path("data/raw_papers")
        
    def scrape_papers(self, subject, topic, max_papers=10):
        """
        Scrape exam papers from CodeChef VIT based on subject and topic.
        
        Args:
            subject (str): The subject area to search for (e.g., "Mathematics", "Physics")
            topic (str): The specific topic within the subject (optional filter)
            max_papers (int): Maximum number of papers to download
            
        Returns:
            list: List of downloaded file paths
        """
        downloaded_files = []
        
        # Create download directory if it doesn't exist
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        print_info(f"Searching CodeChef VIT papers for subject: {subject}")
        
        try:
            # Step 1: Explore the website structure and find papers
            papers_data = self._fetch_all_papers()
            
            if not papers_data:
                print_error("Could not fetch papers from the website")
                return []
            
            # Step 2: Filter papers by subject and topic
            matching_papers = self._filter_papers_by_subject(papers_data, subject, topic)
            
            if not matching_papers:
                print_info(f"No papers found for subject '{subject}' with topic '{topic}'")
                # Show available subjects to help user
                available_subjects = self._get_available_subjects(papers_data)
                if available_subjects:
                    print_info("Available subjects found:")
                    for subj in available_subjects[:10]:
                        print(f"  - {subj}")
                return []
            
            print_success(f"Found {len(matching_papers)} matching papers")
            
            # Step 3: Download papers (limit to max_papers)
            for i, paper_info in enumerate(matching_papers[:max_papers]):
                try:
                    print_progress(f"Downloading paper {i+1}/{min(len(matching_papers), max_papers)}")
                    
                    file_path = self._download_paper(paper_info)
                    if file_path:
                        downloaded_files.append(file_path)
                        print_success(f"✓ {os.path.basename(file_path)}")
                    
                    # Be respectful to the server
                    time.sleep(1)
                    
                except Exception as e:
                    print_error(f"Error downloading paper {i+1}: {str(e)}")
                    continue
                    
        except Exception as e:
            print_error(f"Error during scraping: {str(e)}")
            
        return downloaded_files
    
    def _fetch_all_papers(self):
        """
        Fetch all available papers from CodeChef VIT website.
        This explores the actual website structure.
        """
        papers_data = []
        
        try:
            print_progress("Connecting to CodeChef VIT website...")
            
            # Get main page
            response = self.session.get(self.base_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            print_info("Successfully connected to CodeChef VIT")
            
            # Method 1: Look for paper cards, links, or listings
            # Check for common paper listing patterns
            paper_elements = []
            
            # Look for cards or paper containers
            for selector in ['.paper-card', '.exam-paper', '.paper-item', '.card', 
                           '[class*="paper"]', '[class*="exam"]', '[class*="document"]']:
                elements = soup.select(selector)
                if elements:
                    paper_elements.extend(elements)
                    print_info(f"Found {len(elements)} elements with selector: {selector}")
            
            # Method 2: Look for direct PDF links
            pdf_links = soup.find_all('a', href=True)
            pdf_count = 0
            
            for link in pdf_links:
                href = link.get('href', '')
                
                # Check if it's a PDF link
                if href.endswith('.pdf') or '/download/' in href or '/papers/' in href:
                    full_url = urljoin(self.base_url, href)
                    title = link.get_text().strip() or link.get('title', '') or 'Unknown Paper'
                    
                    # Extract subject information from link or surrounding content
                    subject_info = self._extract_subject_info(link, soup)
                    
                    papers_data.append({
                        'url': full_url,
                        'title': title,
                        'subject_guess': subject_info.get('subject', 'Unknown'),
                        'course_code': subject_info.get('course_code', ''),
                        'year': subject_info.get('year', ''),
                        'text': title,
                        'link_element': str(link)[:200]  # For debugging
                    })
                    pdf_count += 1
            
            print_info(f"Found {pdf_count} potential PDF links")
            
            # Method 3: Look for AJAX endpoints or API calls
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string:
                    # Look for API endpoints in JavaScript
                    api_matches = re.findall(r'["\']([^"\']*(?:api|papers|download)[^"\']*)["\']', 
                                           script.string)
                    for match in api_matches:
                        if 'papers' in match.lower():
                            try:
                                api_papers = self._try_api_endpoint(match)
                                papers_data.extend(api_papers)
                            except:
                                continue
            
            # Method 4: Try common search/browse endpoints
            common_endpoints = [
                '/papers',
                '/browse',
                '/search',
                '/api/papers',
                '/downloads'
            ]
            
            for endpoint in common_endpoints:
                try:
                    endpoint_papers = self._try_endpoint(endpoint)
                    papers_data.extend(endpoint_papers)
                except:
                    continue
            
            print_info(f"Total papers found: {len(papers_data)}")
            return papers_data
            
        except requests.RequestException as e:
            print_error(f"Network error accessing {self.base_url}: {str(e)}")
            print_info("Check your internet connection and website accessibility")
            return []
        except Exception as e:
            print_error(f"Error exploring website: {str(e)}")
            return []
    
    def _extract_subject_info(self, link_element, soup):
        """Extract subject information from link context."""
        subject_info = {'subject': 'Unknown', 'course_code': '', 'year': ''}
        
        # Get text content around the link
        link_text = link_element.get_text().strip().lower()
        
        # Look in parent elements for context
        parent = link_element.parent
        context_text = ""
        
        for _ in range(3):  # Go up 3 levels
            if parent:
                context_text += " " + parent.get_text().strip().lower()
                parent = parent.parent
            else:
                break
        
        combined_text = (link_text + " " + context_text).lower()
        
        # Subject detection patterns
        subject_patterns = {
            'mathematics': ['math', 'calculus', 'algebra', 'geometry', 'statistics', 'probability', 'ma101', 'ma102'],
            'physics': ['physics', 'mechanics', 'thermodynamics', 'optics', 'quantum', 'phy101', 'phy102'],
            'chemistry': ['chemistry', 'organic', 'inorganic', 'physical chemistry', 'che101', 'che102'],
            'computer science': ['computer', 'programming', 'algorithm', 'data structure', 'software', 'cse101', 'cs101'],
            'electronics': ['electronics', 'digital', 'analog', 'microprocessor', 'vlsi', 'ece101', 'ee101'],
            'mechanical': ['mechanical', 'thermodynamics', 'fluid', 'manufacturing', 'me101', 'mech'],
            'civil': ['civil', 'structural', 'construction', 'environmental', 'ce101'],
            'electrical': ['electrical', 'power', 'machines', 'control', 'ee101'],
            'biotechnology': ['bio', 'biotechnology', 'genetics', 'molecular', 'bt101'],
            'chemical': ['chemical engineering', 'process', 'reaction', 'che101']
        }
        
        # Find matching subject
        for subject, keywords in subject_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                subject_info['subject'] = subject.title()
                break
        
        # Extract course code
        course_match = re.search(r'\b([a-z]{2,4}\d{3})\b', combined_text)
        if course_match:
            subject_info['course_code'] = course_match.group(1).upper()
        
        # Extract year
        year_match = re.search(r'\b(20\d{2})\b', combined_text)
        if year_match:
            subject_info['year'] = year_match.group(1)
        
        return subject_info
    
    def _try_api_endpoint(self, endpoint):
        """Try to fetch papers from an API endpoint."""
        papers = []
        
        try:
            if not endpoint.startswith('http'):
                endpoint = urljoin(self.base_url, endpoint)
            
            response = self.session.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                # Try to parse as JSON first
                try:
                    data = response.json()
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'url' in item:
                                papers.append(item)
                    elif isinstance(data, dict) and 'papers' in data:
                        papers.extend(data['papers'])
                except:
                    # Parse as HTML
                    soup = BeautifulSoup(response.content, 'html.parser')
                    pdf_links = soup.find_all('a', href=True)
                    
                    for link in pdf_links:
                        href = link.get('href')
                        if href and '.pdf' in href:
                            papers.append({
                                'url': urljoin(endpoint, href),
                                'title': link.get_text().strip(),
                                'subject_guess': 'Unknown'
                            })
        
        except Exception as e:
            pass  # Silently fail for exploratory requests
        
        return papers
    
    def _try_endpoint(self, endpoint):
        """Try to explore a specific endpoint for papers."""
        papers = []
        
        try:
            url = urljoin(self.base_url, endpoint)
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for PDF links
                pdf_links = soup.find_all('a', href=True)
                for link in pdf_links:
                    href = link.get('href')
                    if href and ('.pdf' in href or '/download/' in href):
                        full_url = urljoin(url, href)
                        title = link.get_text().strip() or 'Paper'
                        subject_info = self._extract_subject_info(link, soup)
                        
                        papers.append({
                            'url': full_url,
                            'title': title,
                            'subject_guess': subject_info.get('subject', 'Unknown'),
                            'course_code': subject_info.get('course_code', ''),
                            'year': subject_info.get('year', ''),
                            'text': title
                        })
                
                time.sleep(0.5)  # Be respectful
        
        except Exception:
            pass  # Silently fail for exploratory requests
        
        return papers
    
    def _filter_papers_by_subject(self, papers_data, subject, topic):
        """Filter papers based on subject and topic with improved matching."""
        if not papers_data:
            return []
        
        filtered_papers = []
        subject_lower = subject.lower()
        topic_lower = topic.lower() if topic else ""
        
        # If searching for "Any", return a sample of all papers
        if subject_lower == "any":
            return papers_data[:max_papers] if papers_data else []
        
        for paper in papers_data:
            paper_text = f"{paper.get('title', '')} {paper.get('text', '')} {paper.get('course_code', '')}".lower()
            subject_guess = paper.get('subject_guess', '').lower()
            
            # Check if paper matches subject
            subject_match = (
                subject_lower in paper_text or 
                subject_lower in subject_guess or
                any(word in paper_text for word in subject_lower.split()) or
                any(word in subject_guess for word in subject_lower.split())
            )
            
            # Special handling for common subject variations
            subject_variations = {
                'computer science': ['computer', 'cse', 'cs', 'programming', 'software'],
                'mathematics': ['math', 'maths', 'ma', 'calculus', 'algebra'],
                'physics': ['physics', 'phy', 'ph', 'mechanics'],
                'electronics': ['electronics', 'ece', 'ee', 'electrical'],
                'mechanical': ['mechanical', 'me', 'mech'],
                'chemistry': ['chemistry', 'che', 'ch', 'chem']
            }
            
            if not subject_match and subject_lower in subject_variations:
                variations = subject_variations[subject_lower]
                subject_match = any(var in paper_text or var in subject_guess for var in variations)
            
            # Check if paper matches topic (if provided)
            topic_match = True
            if topic_lower and topic_lower != "any":
                topic_match = (
                    topic_lower in paper_text or
                    any(word in paper_text for word in topic_lower.split())
                )
            
            if subject_match and topic_match:
                filtered_papers.append(paper)
        
        return filtered_papers
    
    def _get_available_subjects(self, papers_data):
        """Extract available subjects from papers data."""
        subjects = set()
        
        for paper in papers_data:
            subject_guess = paper.get('subject_guess', '').strip()
            if subject_guess and subject_guess != 'Unknown':
                subjects.add(subject_guess)
            
            # Also extract from title
            title = paper.get('title', '').lower()
            
            # Common subject patterns in titles
            if any(word in title for word in ['math', 'calculus', 'algebra']):
                subjects.add('Mathematics')
            elif any(word in title for word in ['physics', 'mechanics']):
                subjects.add('Physics')
            elif any(word in title for word in ['computer', 'programming', 'cse']):
                subjects.add('Computer Science')
            elif any(word in title for word in ['electronics', 'ece', 'digital']):
                subjects.add('Electronics')
            elif any(word in title for word in ['mechanical', 'mech']):
                subjects.add('Mechanical Engineering')
        
        return sorted(list(subjects))
    
    def _download_paper(self, paper_info):
        """
        Download a paper if it doesn't already exist.
        
        Args:
            paper_info (dict): Paper information including URL and title
            
        Returns:
            str: Path to downloaded file or None if failed
        """
        # Generate safe filename
        title = paper_info.get('title', 'unknown_paper')
        course_code = paper_info.get('course_code', '')
        year = paper_info.get('year', '')
        
        # Create descriptive filename
        filename_parts = []
        if year:
            filename_parts.append(year)
        if course_code:
            filename_parts.append(course_code)
        
        # Clean title
        clean_title = re.sub(r'[<>:"/\\|?*]', '', title)
        clean_title = re.sub(r'\s+', '_', clean_title)
        filename_parts.append(clean_title[:50])  # Limit title length
        
        filename = "_".join(filename_parts) + ".pdf"
        filename = filename.replace('__', '_').strip('_')
        
        file_path = self.download_dir / filename
        
        # Check if file already exists
        if file_path.exists():
            print_info(f"File already exists: {filename}")
            return str(file_path)
        
        try:
            response = self.session.get(paper_info['url'], timeout=30, stream=True)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            
            # Read first chunk to check PDF signature
            first_chunk = next(response.iter_content(chunk_size=1024), b'')
            if not first_chunk.startswith(b'%PDF'):
                print_error(f"URL does not contain a valid PDF: {paper_info['url']}")
                return None
            
            # Save the file
            with open(file_path, 'wb') as f:
                f.write(first_chunk)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = file_path.stat().st_size
            if file_size < 1000:  # Less than 1KB is suspicious
                print_error(f"Downloaded file seems too small: {filename}")
                file_path.unlink()  # Delete the file
                return None
            
            print_success(f"Downloaded: {filename} ({file_size:,} bytes)")
            return str(file_path)
            
        except Exception as e:
            print_error(f"Failed to download {paper_info['url']}: {str(e)}")
            return None
    
    def list_available_subjects(self):
        """
        Get a list of available subjects from the CodeChef VIT website.
        """
        try:
            print_progress("Exploring CodeChef VIT website for subjects...")
            
            # Fetch all papers
            papers_data = self._fetch_all_papers()
            
            if not papers_data:
                print_error("Could not fetch data from website")
                return []
            
            # Extract available subjects
            subjects = self._get_available_subjects(papers_data)
            
            if subjects:
                print_success(f"Found {len(subjects)} subjects")
                return subjects
            else:
                print_info("No subjects detected. The website might have a different structure.")
                # Return some example subjects to guide user
                return ["Mathematics", "Physics", "Computer Science", "Electronics", "Mechanical Engineering"]
                
        except Exception as e:
            print_error(f"Error fetching subjects: {str(e)}")
            return []
    
    def debug_website_structure(self):
        """Debug function to understand the website structure."""
        try:
            print_info("Debugging CodeChef VIT website structure...")
            
            response = self.session.get(self.base_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            print_info("Page title: " + (soup.title.string if soup.title else "No title"))
            
            # Check for common elements
            print_info("Found elements:")
            for tag in ['nav', 'menu', 'header', 'main', 'section', 'div']:
                elements = soup.find_all(tag)
                if elements:
                    print_info(f"  {tag}: {len(elements)} elements")
            
            # Check for classes that might contain papers
            all_classes = []
            for element in soup.find_all(class_=True):
                all_classes.extend(element.get('class'))
            
            unique_classes = set(all_classes)
            paper_related_classes = [cls for cls in unique_classes 
                                   if any(word in cls.lower() for word in ['paper', 'exam', 'document', 'card', 'item'])]
            
            if paper_related_classes:
                print_info(f"Paper-related classes found: {paper_related_classes}")
            
            # Look for links
            all_links = soup.find_all('a', href=True)
            pdf_links = [link for link in all_links if '.pdf' in link.get('href', '')]
            
            print_info(f"Total links: {len(all_links)}")
            print_info(f"PDF links: {len(pdf_links)}")
            
            if pdf_links:
                print_info("Sample PDF links:")
                for link in pdf_links[:3]:
                    print_info(f"  - {link.get('href')} ({link.get_text().strip()[:50]})")
            
        except Exception as e:
            print_error(f"Debug failed: {str(e)}")


if __name__ == "__main__":
    # Test the scraper with debug mode
    scraper = ExamScraper()
    
    print("=== DEBUG MODE ===")
    scraper.debug_website_structure()
    
    print("\n=== TESTING SUBJECT LISTING ===")
    subjects = scraper.list_available_subjects()
    if subjects:
        print_info("Available subjects:")
        for subject in subjects[:10]:
            print(f"  - {subject}")
    
    print("\n=== TESTING PAPER SCRAPING ===")
    papers = scraper.scrape_papers("Computer Science", "Programming", max_papers=2)
    print(f"Downloaded {len(papers)} papers:")
    for paper in papers:
        print(f"  • {paper}")