import scrapy
from scrapy.crawler import CrawlerProcess
from duckduckgo_search import DDGS
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.http import HtmlResponse
import json
from typing import List, Dict, Set
from scrapy import signals
from scrapy.signalmanager import dispatcher
from bs4 import BeautifulSoup
import re


class DDGItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    text = scrapy.Field()
    html = scrapy.Field()
    search_query = scrapy.Field()


class DDGCrawlSpider(CrawlSpider):
    name = "ddg_crawl_spider"
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS': 8,
    }

    # def limit_links(self, links):
    #     # Limit to the first 5 links
    #     print(f"Found {len(links)} links but limitng to 10 links")
    #     print(f"Full links are {links}")
    #     # print(f"limited links are {links[:10]}")
    #     return links
    
    # rules = (
    #     Rule(LinkExtractor(allow_domains=[]), callback='parse_page', follow=False,process_links='limit_links'),
    # )
    rules = (
        Rule(LinkExtractor(allow_domains=[],
                           deny=(
                r'privacy-policy',       # Privacy Policy
                r'terms-of-service',     # Terms of Service
                r'terms-and-conditions', # Terms and Conditions
                r'terms',                # Terms and Conditions
                r'career',               # career pages
                r'job',                  # job pages
                r'cookie-policy',        # Cookie Policy
                r'legal',                # Legal
                r'cancel',               # Cancel pages
                r'refund',               # Refund pages
                r'login',                # Login pages
                r'join',                 # join pages
                r'logout',               # Logout pages
                r'register',             # register pages
                r'sign-in',              # Sign-in pages
                r'authenticate',         # Authentication pages
                r'admin',                # Admin pages
                r'backend',              # Backend pages
                r'dashboard',            # Dashboard pages
                r'privacy-policy',       # Another common name for Privacy Policy
                r'404',                  # 404 Error pages
                r'not-found',            # Not found pages
                r'.(css|js|jpg|jpeg|png|gif|svg|ico)$',  # Static resources like images, JS, CSS
                r'/search',              # Search result pages
                r'/query',               # Query pages
                r'/results',             # Results pages
                r'/page=',               # Paginated URLs
                r'/?page=',              # Paginated URLs (common query parameter)
                r'/?p=',                 # Another paginated URL format
                r'\?session_id=',        # Session tracking URLs
                r'\?tracking=',          # Tracking parameters in URLs
                r'\?ref=',               # Referral parameters in URLs
                r'/comments/',           # Comments pages
                r'/tags/',               # Tags pages
                r'/filter=',             # Filter pages
                r'/product/.*?/',        # Product variant URLs (e.g., colors, sizes)
                r'#',                    # URL fragments (anchors)
                r'/share/',              # Social share URLs
                r'/facebook.com/',       # Social media URLs
                r'/twitter.com/',        # Social media URLs
                r'/instagram.com/',      # Social media URLs
                r'/linkedin.com/',       # Social media URLs
                r'/youtube.com/',        # Social media URLs
                r'sitemap.xml',          # SiteMap
            )
                        ), 
                           callback='parse_page', follow=False),
    )
    
    def __init__(self, query=None,url_list = [], max_results=10, *args, **kwargs):
        super(DDGCrawlSpider, self).__init__(*args, **kwargs)
        self.query = query
        self.max_results = max_results
        self.ddgs = DDGS()
        self.results = []
        self.allowed_domains = []
        self.processed_urls = set()
        # self.start_urls = self.get_search_results()
        self.start_urls = url_list

    def parse_start_url(self, response):
        print(f"From parse start url url is {response.url}")
        return self.parse_page(response)
    
    def get_search_results(self):
        print(f"Searching for query: {self.query} and to get {self.max_results} results")
        search_results = self.ddgs.text(self.query, max_results=self.max_results,backend='json')
        urls = [result['href'] for result in search_results]
        # self.allowed_domains = list(set([self.extract_domain(url) for url in urls]))
        print(f"Allowed domains: {self.allowed_domains}")
        print(f"Allowed urls: {urls}")
        print(f"Duck Duck Go Search size is : {len(urls)}")
        return urls
    
    def extract_domain(self, url):
        return url.split("/")[2] if "//" in url else url
    
    def parse_page(self, response):
        if response.url in self.processed_urls:
            print(f"Already processed URL: {response.url}")
            return
        self.processed_urls.add(response.url)
        item = DDGItem()
        print(f"From parse page url is {response.url}")
        item['url'] = response.url
        item['title'] = response.xpath('//title/text()').get()
        item['text'] = self.extract_text(response.text)
        item['html'] = response.text
        item['search_query'] = self.query
        
        self.results.append(dict(item))
        yield item
    
    def extract_text(self, response):
        # text = ' '.join(response.xpath('//body//text()').extract())
        # import re
        # text = re.sub(r'\s+', ' ', text).strip()
        # return text
        """Extract plain text from HTML content"""
        if not response:
            return ""
            
        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(response, 'html.parser')
        
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


def run_ddg_crawl_spider(query, max_results=10,url_list=[], output_file=None):
    results = []
    process_settings = {'LOG_LEVEL': 'INFO'}
    if output_file:
        process_settings['FEEDS'] = {output_file: {'format': 'json', 'overwrite': True}}
    process = CrawlerProcess(process_settings)
    spider_results = []
    
    def collect_item(item, response, spider):
        spider_results.append(dict(item))
    
    dispatcher.connect(collect_item, signal=signals.item_scraped)
    process.crawl(DDGCrawlSpider, query=query, url_list=url_list,max_results=max_results)
    process.start()
    return spider_results


class EnhancedScraper:
    def __init__(self):
        self.ddgs = DDGS()
        self.urls = []
    
    def search(self, query, max_results=10):
        search_results =  self.ddgs.text(query, max_results=max_results)
        # self.allowed_domains = list(set([self.extract_domain(url) for url in urls]))
        urls = [result['href'] for result in search_results]
        print(f"Allowed urls: {urls}")
        print(f"Duck Duck Go Search size is : {len(urls)}")
        return urls
    
    def scrape(self, query, max_results=10, output_file=None):
        self.urls = self.search(query, max_results)
        results = run_ddg_crawl_spider(query, max_results,self.urls, output_file)
        # return {
        #     'urls': [r['url'] for r in results],
        #     'titles': [r['title'] for r in results],
        #     'extracted_texts': [r['text'] for r in results],
        #     'html_content': [r['html'] for r in results],
        #     'full_results': results
        # }
        return results


if __name__ == "__main__":
    scraper = EnhancedScraper()
    results = scraper.scrape("Jupudi Industries", max_results=10, output_file="crawl_results_final_new.json")
    print(f"Scraped {len(results)} URLs:")
    for item in results:
        print(item['url'])
        print(item['title'])
        print("----")