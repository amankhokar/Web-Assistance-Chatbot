"""
Web Crawler Module
Handles website content extraction and cleaning
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
import html2text
from urllib.parse import urljoin, urlparse
import validators
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebCrawler:
    """
    Crawls websites and extracts clean text content
    Removes headers, footers, navigation, ads, and other irrelevant content
    """
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        
    def validate_url(self, url: str) -> bool:
        """Validate if URL is properly formatted and reachable"""
        if not validators.url(url):
            return False
        return True
    
    def fetch_page(self, url: str) -> Optional[requests.Response]:
        """
        Fetch webpage content with error handling
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch URL {url}: {str(e)}")
            return None
    
    def remove_irrelevant_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """
        Remove headers, footers, navigation, ads, scripts, styles
        """
        # Tags to remove completely
        tags_to_remove = [
            'header', 'footer', 'nav', 'aside', 
            'script', 'style', 'noscript', 'iframe',
            'form', 'button'
        ]
        
        for tag in tags_to_remove:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove by class/id patterns (common for ads and navigation)
        irrelevant_patterns = [
            'nav', 'menu', 'sidebar', 'header', 'footer',
            'advertisement', 'ads', 'banner', 'cookie',
            'popup', 'modal', 'social', 'share'
        ]
        
        for pattern in irrelevant_patterns:
            for element in soup.find_all(class_=lambda x: x and pattern in x.lower()):
                element.decompose()
            for element in soup.find_all(id=lambda x: x and pattern in x.lower()):
                element.decompose()
        
        return soup
    
    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, str]:
        """Extract page title and metadata"""
        title = "Unknown Title"
        
        # Try to get title
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        
        return {
            'title': title,
            'url': url,
            'domain': urlparse(url).netloc
        }
    
    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from cleaned HTML
        Focus on article, main, div with content classes
        """
        # Priority tags for main content
        main_content_tags = ['article', 'main', {'class': 'content'}, 
                            {'class': 'post'}, {'class': 'entry'}]
        
        main_content = None
        for tag in main_content_tags:
            if isinstance(tag, dict):
                main_content = soup.find('div', tag)
            else:
                main_content = soup.find(tag)
            if main_content:
                break
        
        # Fallback to body if no main content found
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            return main_content.get_text(separator=' ', strip=True)
        return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        Remove extra whitespace, empty lines, special characters
        """
        # Remove extra whitespace
        lines = [line.strip() for line in text.split('\n')]
        # Remove empty lines
        lines = [line for line in lines if line]
        # Join with single newline
        cleaned = '\n'.join(lines)
        
        # Remove multiple spaces
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def crawl(self, url: str) -> Optional[Dict]:
        """
        Main crawling function
        Returns: Dict with content and metadata or None if failed
        """
        # Validate URL
        if not self.validate_url(url):
            logger.error(f"Invalid URL: {url}")
            return None
        
        # Fetch page
        response = self.fetch_page(url)
        if not response:
            return None
        
        # Parse HTML
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to parse HTML: {str(e)}")
            return None
        
        # Extract metadata
        metadata = self.extract_metadata(soup, url)
        
        # Remove irrelevant content
        soup = self.remove_irrelevant_content(soup)
        
        # Extract main content
        raw_text = self.extract_main_content(soup)
        
        # Clean text
        clean_text = self.clean_text(raw_text)
        
        if not clean_text or len(clean_text) < 100:
            logger.warning(f"Insufficient content extracted from {url}")
            return None
        
        logger.info(f"Successfully crawled {url} - Extracted {len(clean_text)} characters")
        
        return {
            'content': clean_text,
            'metadata': metadata
        }


if __name__ == "__main__":
    # Test the crawler
    crawler = WebCrawler()
    result = crawler.crawl("https://en.wikipedia.org/wiki/Artificial_intelligence")
    if result:
        print(f"Title: {result['metadata']['title']}")
        print(f"Content length: {len(result['content'])} characters")
        print(f"First 500 chars:\n{result['content'][:500]}")
