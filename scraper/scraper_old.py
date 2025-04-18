from duckduckgo_search import DDGS
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig
import json
from crawl4ai.models import CrawlResult
from bs4 import BeautifulSoup
import re





class Scraper:

    def __init__(self):
        self.browser_cfg = BrowserConfig(headless=True)
        self.ddgs = DDGS()
        
    
    def search(self,query, max_results=10):
        return self.ddgs.text(query, max_results=max_results)
    
    async def get_scraped_results(self,urls):
        async with AsyncWebCrawler(browser_config=self.browser_cfg) as crawler:
            results = await crawler.arun_many(urls)
        return results
    
    def scrape(self, query, max_results=10):
        urls=[]
        titles = []
        body_list = []
        result_content = []
        search_results = self.search(query, max_results)
        for url in search_results:
            urls.append(url['href'])
            titles.append(url['title'])
            body_list.append(url['body'])
        results = asyncio.run(self.get_scraped_results(urls))
        # print(results[0])
        for result in results:
            res:CrawlResult = result
            extracted_text = self.extract_text_from_html(res.html)
            result_content.append(extracted_text)

        return result_content[0]
    
    def extract_text_from_html(self, html_content):
        """Extract plain text from HTML content"""
        if not html_content:
            return ""
            
        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

    

if __name__ == '__main__':
    scraper = Scraper()
    query = 'Jupudi Industries'
    results = scraper.scrape(query)
    print(results)
    
    
