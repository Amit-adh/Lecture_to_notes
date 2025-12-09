"""
Scraper for CodeChef VIT exam papers.
Uses requests by default; if Playwright is available it will be used to render dynamic pages.
"""
import os
import re
import time
from typing import List, Optional
from urllib.parse import urljoin

from pathlib import Path
from bs4 import BeautifulSoup

from m import print_info, print_success, print_error, print_progress, clean_filename

# Try to import Playwright if available (optional)
try:
    from playwright.sync_api import sync_playwright
    _HAS_PLAYWRIGHT = True
except Exception:
    _HAS_PLAYWRIGHT = False


class ExamScraper:
    def __init__(self):
        import requests
        self.requests = requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.base_url = "https://papers.codechefvit.com"
        self.download_dir = Path("data/raw_papers")

    def _fetch_via_playwright(self, url: str, subject: Optional[str] = None) -> str:
        if not _HAS_PLAYWRIGHT:
            raise RuntimeError("Playwright not available")
        print_info("Starting Playwright browser...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  # Set to False to see what's happening
            page = browser.new_page()
            print_info(f"Navigating to {url}...")
            page.goto(url, timeout=30000)
            
            if subject:
                print_info("Looking for search input...")
                # First check what input elements are available
                inputs = page.query_selector_all('input')
                for inp in inputs:
                    placeholder = inp.get_attribute('placeholder') or ''
                    print_info(f"Found input with placeholder: {placeholder}")
                
                # Try different search input selectors
                search_selectors = [
                    '[placeholder="Search by subject..."]',
                    '[type="search"]',
                    'input[placeholder*="search" i]',
                    'input[type="text"]'
                ]
                
                search_input = None
                for selector in search_selectors:
                    print_info(f"Trying selector: {selector}")
                    try:
                        search_input = page.wait_for_selector(selector, timeout=5000)
                        if search_input:
                            print_success(f"Found search input with selector: {selector}")
                            break
                    except:
                        continue
                
                if search_input:
                    print_info(f"Typing search term: {subject}")
                    # Clear any existing text first
                    search_input.click()
                    page.keyboard.press("Control+A")
                    page.keyboard.press("Delete")
                    search_input.type(subject, delay=100)  # Type slower to ensure it registers
                    page.keyboard.press("Enter")  # Try pressing enter to trigger search
                    
                    # Wait for potential loading states
                    print_info("Waiting for results...")
                    time.sleep(2)
                    
                    # Look for and click the subject link
                    subject_links = page.query_selector_all('a')
                    subject_clicked = False
                    for link in subject_links:
                        text = link.text_content() or ''
                        if subject.lower() in text.lower():
                            print_info(f"Found subject link: {text}")
                            link.click()
                            subject_clicked = True
                            print_success("Clicked on subject link")
                            time.sleep(2)  # Wait for page to load
                            break
                    
                    if not subject_clicked:
                        print_error(f"Could not find link for subject: {subject}")
                    
                    # Now look for paper links (they will have /paper/ in the URL)
                    print_info("Looking for paper links...")
                    paper_links = page.query_selector_all('a[href*="/paper/"]')
                    found_papers = []
                    for link in paper_links:
                        href = link.get_attribute('href')
                        text = link.text_content()
                        if href and text:
                            full_url = urljoin(url, href)
                            print_info(f"Found paper: {text} -> {href}")
                            found_papers.append({
                                'url': full_url,
                                'title': text,
                                'text': text,
                                'subject': subject
                            })
                            
                else:
                    print_error("Could not find search input")
                    
            # Take a screenshot for debugging
            page.screenshot(path="debug_screenshot.png")
            print_info("Saved debug screenshot to debug_screenshot.png")
            
            return page.content()
            
            content = page.content()
            browser.close()
            return content

    def _fetch_via_requests(self, url: str) -> str:
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text

    def _get_page(self, url: str) -> str:
        try:
            if _HAS_PLAYWRIGHT:
                print_progress("Using Playwright to fetch page")
                return self._fetch_via_playwright(url)
            else:
                return self._fetch_via_requests(url)
        except Exception as e:
            print_error(f"Failed to fetch {url}: {e}")
            return ""

    def _fetch_all_papers(self, subject: Optional[str] = None) -> List[dict]:
        papers = []
        
        # Use Playwright for search functionality
        if subject and _HAS_PLAYWRIGHT:
            html = self._fetch_via_playwright(self.base_url, subject)
        else:
            html = self._get_page(self.base_url)
            
        if not html:
            return papers
            
        soup = BeautifulSoup(html, 'html.parser')

        # Find paper cards/containers
        print_info("Looking for paper containers...")
        paper_cards = soup.find_all('div', {'class': 'paper-card'})
        if paper_cards:
            print_info(f"Found {len(paper_cards)} paper cards")
        
        # Look for any paper links regardless
        print_info("Looking for PDF links...")
        pdf_count = 0
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(separator=' ', strip=True)
            if not href:
                continue
            if href.lower().endswith('.pdf') or '.pdf' in href.lower() or '/paper/' in href.lower():
                pdf_count += 1
                full = urljoin(self.base_url, href)
                papers.append({
                    'url': full,
                    'title': text or os.path.basename(href),
                    'text': text,
                    'subject': subject or 'Unknown'
                })
                print_info(f"Found PDF: {text} -> {href}")
        
        print_info(f"Total PDFs found: {pdf_count}")
        
        # Also print some sample links to see what's available
        print_info("\nSample links found on page:")
        for a in soup.find_all('a', href=True)[:5]:
            href = a['href']
            text = a.get_text(separator=' ', strip=True)
            print_info(f"Link: {text} -> {href}")
        return papers

    def _extract_subject_info(self, link_element, soup):
        # Simple heuristic: look at nearby text for course code/year
        subj = 'Unknown'
        course_code = ''
        year = ''
        parent = link_element.parent
        context = ''
        for _ in range(3):
            if parent:
                context += ' ' + parent.get_text(separator=' ', strip=True)
                parent = parent.parent
        txt = (link_element.get_text(separator=' ', strip=True) + ' ' + context).lower()
        # course code like cse101
        m = re.search(r'\b([a-z]{2,4}\d{3})\b', txt)
        if m:
            course_code = m.group(1).upper()
        y = re.search(r'\b(20\d{2})\b', txt)
        if y:
            year = y.group(1)
        # quick subject guesses
        if any(k in txt for k in ['math', 'algebra', 'calculus']):
            subj = 'Mathematics'
        elif any(k in txt for k in ['physics', 'mechanics']):
            subj = 'Physics'
        elif any(k in txt for k in ['computer', 'cse', 'programming']):
            subj = 'Computer Science'
        return {'subject': subj, 'course_code': course_code, 'year': year}

    def _filter_papers_by_subject(self, papers, subject, topic):
        if not papers:
            return []
        s = subject.lower()
        t = (topic or '').lower()
        if s == 'any':
            return papers
        out = []
        for p in papers:
            txt = (p.get('title','') + ' ' + p.get('text','')).lower()
            match_sub = s in txt or any(w in txt for w in s.split())
            match_topic = True
            if t:
                match_topic = t in txt or any(w in txt for w in t.split())
            if match_sub and match_topic:
                out.append(p)
        return out

    def _download_paper(self, paper_info: dict) -> Optional[str]:
        if not paper_info.get('url'):
            return None
        url = paper_info['url']
        
        try:
            # If it's a paper page (not direct PDF), we need to visit it first
            if '/paper/' in url.lower():
                print_info(f"Visiting paper page to get PDF: {url}")
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, timeout=30000)
                    # Wait for potential download button or links to render
                    time.sleep(2)

                    # First try to find any direct PDF link in the rendered DOM
                    pdf_elem = page.query_selector('a[href$=".pdf"], a[href*=".pdf"]')
                    if pdf_elem:
                        pdf_url = pdf_elem.get_attribute('href')
                        if pdf_url:
                            url = urljoin(self.base_url, pdf_url)
                            print_success(f"Found PDF download link in page: {url}")
                            browser.close()
                    else:
                        # Try common selectors that may trigger a download when clicked
                        download_selectors = [
                            'a[class*="download"]',
                            'a[title*="Download"]',
                            'a[aria-label*="download"]',
                            'button[class*="download"]',
                            'button[title*="Download"]',
                            'button[aria-label*="download"]',
                            'a[href*="/download"], a[href*="/paper/"]',
                        ]

                        found = False
                        for sel in download_selectors:
                            try:
                                btn = page.query_selector(sel)
                                if not btn:
                                    continue
                                # Attempt to capture a download event if clicking triggers one
                                try:
                                    with page.expect_download(timeout=5000) as dl_info:
                                        btn.click()
                                    download = dl_info.value
                                    # Save to a temporary path; will be saved again later
                                    tmp_path = str(self.download_dir / (download.suggested_filename or 'paper.pdf'))
                                    download.save_as(tmp_path)
                                    print_success(f"Captured download via click: {tmp_path}")
                                    # Use the saved file as the final result
                                    browser.close()
                                    return tmp_path
                                except Exception:
                                    # If no download event, perhaps the click revealed an anchor
                                    time.sleep(1)
                                    pdf_elem = page.query_selector('a[href$=".pdf"], a[href*=".pdf"]')
                                    if pdf_elem:
                                        pdf_url = pdf_elem.get_attribute('href')
                                        if pdf_url:
                                            url = urljoin(self.base_url, pdf_url)
                                            print_success(f"Found PDF link after clicking: {url}")
                                            found = True
                                            break
                            except Exception:
                                continue

                        if not found:
                            # As a last resort, try to extract any PDF URL embedded in page scripts
                            try:
                                page_html = page.content()
                                m = re.search(r'https://storage\.googleapis\.com/[^"\']+?\.pdf', page_html)
                                if m:
                                    url = m.group(0)
                                    print_success(f"Extracted PDF URL from page scripts: {url}")
                                    try:
                                        browser.close()
                                    except Exception:
                                        pass
                                    # proceed to download the extracted URL below
                                    found = True
                            except Exception:
                                pass

                        if not found:
                            print_error("Could not find PDF download link on paper page")
                            # Save debug artifacts for inspection
                            try:
                                safe_name = clean_filename(os.path.basename(url)) or 'paper'
                                screenshot_path = f"paper_debug_{safe_name}.png"
                                html_path = f"paper_debug_{safe_name}.html"
                                page.screenshot(path=screenshot_path)
                                with open(html_path, 'w', encoding='utf-8') as hf:
                                    hf.write(page.content())
                                print_info(f"Saved paper debug screenshot: {screenshot_path}")
                                print_info(f"Saved paper debug HTML: {html_path}")
                            except Exception:
                                pass
                            try:
                                browser.close()
                            except Exception:
                                pass
                            return None
            
            # Now try to download the actual PDF
            fname = clean_filename(paper_info.get('title', '')) + '.pdf'
            path = self.download_dir / fname
            
            if path.exists():
                print_info(f"Paper already exists: {path}")
                return str(path)
                
            print_progress(f"Downloading {fname}...")
            
            # First try with requests
            try:
                with open(path, 'wb') as f:
                    resp = self.session.get(url, stream=True, timeout=30)
                    resp.raise_for_status()
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print_success(f"Saved paper to {path}")
                return str(path)
            except Exception as e:
                print_error(f"Failed to download with requests: {e}")
                # If requests fails, try with Playwright
                try:
                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=True)
                        page = browser.new_page()
                        
                        # Create a download promise before navigating
                        with page.expect_download() as download_info:
                            page.goto(url, timeout=30000)
                            time.sleep(1)
                            
                            # Try clicking any download button that appears
                            download_button = page.query_selector('button.download-btn, a.download-btn')
                            if download_button:
                                download_button.click()
                        
                        download = download_info.value
                        print_info(f"Starting download of {download.suggested_filename}")
                        download.save_as(path)
                        
                        browser.close()
                        
                        print_success(f"Saved paper to {path}")
                        return str(path)
                except Exception as e2:
                    print_error(f"Failed to download with Playwright: {e2}")
                    if path.exists():
                        path.unlink()  # Clean up partial download
                    return None
            r.raise_for_status()
            ct = r.headers.get('content-type','').lower()
            if 'pdf' not in ct and not url.lower().endswith('.pdf'):
                print_error(f"URL did not look like a PDF: {url}")
                return None
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            print_success(f"Downloaded: {dest.name}")
            return str(dest)
        except Exception as e:
            print_error(f"Download failed: {e}")
            return None

    def list_available_subjects(self) -> List[str]:
        papers = self._fetch_all_papers()
        subs = set()
        for p in papers:
            title = p.get('title','').lower()
            if 'math' in title:
                subs.add('Mathematics')
            if 'physics' in title:
                subs.add('Physics')
            if 'computer' in title or 'cse' in title:
                subs.add('Computer Science')
        return sorted(list(subs))

    def scrape_papers(self, subject: str, topic: str = None, max_papers: int = 10) -> List[str]:
        print_info(f"Searching for papers in subject: {subject}")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the subject-based search
        papers = self._fetch_all_papers(subject)
        if not papers:
            print_error('No papers discovered for the subject')
            return []
            
        # Apply topic filter if provided
        if topic:
            filtered = self._filter_papers_by_subject(papers, 'any', topic)
        else:
            filtered = papers
            
        if not filtered:
            print_info('No matching papers after filtering')
            return []
            
        print_success(f"Found {len(filtered)} matching papers")
        results = []
        for p in filtered[:max_papers]:
            fp = self._download_paper(p)
            if fp:
                results.append(fp)
                print_success(f"Downloaded paper: {os.path.basename(fp)}")
            time.sleep(1)  # Increased delay to be more respectful
        return results

    def debug_website_structure(self, subject: Optional[str] = None):
        if subject:
            print_info(f"Debugging website structure with subject search: {subject}")
            html = self._fetch_via_playwright(self.base_url, subject) if _HAS_PLAYWRIGHT else self._get_page(self.base_url)
        else:
            print_info("Debugging website structure (homepage)")
            html = self._get_page(self.base_url)
            
        if not html:
            print_error('Could not fetch page')
            return
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for search input
        search_inputs = soup.find_all('input', {'placeholder': True})
        print_info('Search inputs found:')
        for inp in search_inputs:
            print_info(f"  {inp.get('placeholder')} -> {inp.get('class', [])}")
            
        # Look for paper cards/containers
        print_info('\nPossible paper containers:')
        for div in soup.find_all('div', {'class': True})[:10]:
            classes = ' '.join(div.get('class', []))
            if any(term in classes.lower() for term in ['paper', 'card', 'result']):
                print_info(f"  {classes} -> {div.get_text(strip=True)[:60]}")
                
        # Look for PDF links
        print_info('\nPDF links found:')
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '.pdf' in href.lower():
                print_info(f"  {a.get_text(strip=True)[:60]} -> {href}")


if __name__ == '__main__':
    s = ExamScraper()
    s.debug_website_structure()
    subs = s.list_available_subjects()
    print('Subjects:', subs)
    papers = s.scrape_papers('Computer Science', 'Programming', max_papers=2)
    print('Downloaded:', papers)
