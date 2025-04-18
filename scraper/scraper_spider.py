# -- ddg_spider.py --
import scrapy
from scrapy.crawler import CrawlerProcess
from duckduckgo_search import DDGS
from scrapy.linkextractors import LinkExtractor
from scrapy.http import HtmlResponse
import json
import asyncio
from typing import List, Dict, Any
from scrapy import signals
from scrapy.signalmanager import dispatcher


class DDGItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    text = scrapy.Field()
    html = scrapy.Field()
    search_query = scrapy.Field()


class DDGSpider(scrapy.Spider):
    name = "ddg_spider"
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS': 8,
    }
    
    def __init__(self, query=None, max_results=10, *args, **kwargs):
        super(DDGSpider, self).__init__(*args, **kwargs)
        self.query = query
        self.max_results = max_results
        self.ddgs = DDGS()
        self.results = []
    
    def start_requests(self):
        # Get search results from DDG
        search_results = self.ddgs.text(self.query, max_results=self.max_results)
        self.logger.info(f"Found {len(search_results)} search results for query: {self.query}")
        
        # Generate requests for each URL
        for result in search_results:
            url = result['href']
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                meta={
                    'title': result['title'],
                    'description': result['body'],
                    'search_result': result
                },
                errback=self.errback_handler
            )
    
    def parse(self, response):
        # Create item from response
        item = DDGItem()
        item['url'] = response.url
        item['title'] = response.meta.get('title')
        item['text'] = self.extract_text(response)
        item['html'] = response.text
        item['search_query'] = self.query
        
        self.results.append(dict(item))
        yield item
    
    def extract_text(self, response):
        # Extract main text content from response
        text = ' '.join(response.xpath('//body//text()').extract())
        # Clean up the text (remove extra whitespace)
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def errback_handler(self, failure):
        # Handle request failures
        request = failure.request
        self.logger.error(f"Error on {request.url}: {repr(failure)}")
        
        # Create a partial item with error information
        item = DDGItem()
        item['url'] = request.url
        item['title'] = request.meta.get('title')
        item['text'] = f"ERROR: {repr(failure)}"
        item['html'] = ""
        item['search_query'] = self.query
        
        self.results.append(dict(item))
        yield item


def run_ddg_spider(query, max_results=10, output_file=None):
    """
    Run the DDG spider and return the results
    
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to fetch
        output_file (str): Optional path to save results JSON
        
    Returns:
        List[Dict]: The scraped results
    """
    # Create a list to store results
    results = []
    
    # Create process settings
    process_settings = {
        'LOG_LEVEL': 'INFO',
    }
    
    # Add feed export if output file is specified
    if output_file:
        process_settings['FEEDS'] = {
            output_file: {
                'format': 'json',
                'overwrite': True
            }
        }
    
    # Create crawler process
    process = CrawlerProcess(process_settings)
    
    # Store results from spider
    spider_results = []
    
    # Create a callback to collect items
    def collect_item(item, response, spider):
        spider_results.append(dict(item))
    
    # Connect signal
    dispatcher.connect(collect_item, signal=signals.item_scraped)
    
    # Add the spider to the process with the query parameters
    process.crawl(DDGSpider, query=query, max_results=max_results)
    
    # Start the process and block until it's finished
    process.start()  # This is a blocking call
    
    return spider_results


# -- Updated Scraper class integration --
class EnhancedScraper:
    """Enhanced scraper that uses Scrapy for web crawling"""
    
    def __init__(self):
        self.ddgs = DDGS()
    
    def search(self, query, max_results=10):
        return self.ddgs.text(query, max_results=max_results)
    
    def scrape(self, query, max_results=10, output_file=None):
        """
        Scrape search results using Scrapy
        
        Args:
            query (str): The search query
            max_results (int): Maximum number of results to fetch
            output_file (str): Optional path to save results as JSON
            
        Returns:
            dict: The scraped results with urls, titles, texts, and html content
        """
        results = run_ddg_spider(query, max_results, output_file)
        
        # Organize the results
        urls = [r['url'] for r in results]
        titles = [r['title'] for r in results]
        texts = [r['text'] for r in results]
        html_content = [r['html'] for r in results]
        
        return {
            'urls': urls,
            'titles': titles,
            'extracted_texts': texts,
            'html_content': html_content,
            'full_results': results
        }


# -- Example usage --
if __name__ == "__main__":
    scraper = EnhancedScraper()
    results = scraper.scrape("Jupudi Industries", max_results=10, output_file="python_results.json")
    
    print(f"Scraped {len(results['urls'])} URLs:")
    for i, url in enumerate(results['urls']):
        print(f"{i+1}. {url} - {results['titles'][i]}")
        print(f"   Text length: {len(results['extracted_texts'][i])} characters")
        print("-" * 50)