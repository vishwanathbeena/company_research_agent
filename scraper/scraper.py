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
import random
import time
from scrapy.downloadermiddlewares.useragent import UserAgentMiddleware
from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.utils.response import response_status_message


# LinkedIn-specific middleware to handle 999 errors
class LinkedInMiddleware(RetryMiddleware):
    """Middleware to handle LinkedIn's anti-scraping measures"""
    
    # List of user agents to rotate
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15'
    ]
    
    def __init__(self, settings):
        super(LinkedInMiddleware, self).__init__(settings)
        self.max_delay = settings.getfloat('LINKEDIN_MAX_DELAY', 7.0)
        self.min_delay = settings.getfloat('LINKEDIN_MIN_DELAY', 3.0)
        self.debug = settings.getbool('LINKEDIN_DEBUG', True)
        
    @classmethod
    def from_crawler(cls, crawler):
        middleware = cls(crawler.settings)
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        return middleware
    
    def spider_opened(self, spider):
        spider.logger.info('LinkedInMiddleware: initialized')
        
    def process_request(self, request, spider):
        # Only apply special handling to LinkedIn URLs
        if 'linkedin.com' not in request.url:
            return None
            
        # Set a random user agent
        user_agent = random.choice(self.user_agents)
        request.headers['User-Agent'] = user_agent
        
        # Add common headers that real browsers would send
        request.headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
        request.headers['Accept-Language'] = 'en-US,en;q=0.9'
        request.headers['Accept-Encoding'] = 'gzip, deflate, br'
        request.headers['Connection'] = 'keep-alive'
        request.headers['Upgrade-Insecure-Requests'] = '1'
        
        # Set referer to make it look like you're coming from a LinkedIn page
        request.headers['Referer'] = 'https://www.linkedin.com/'
        
        # Random delay before each LinkedIn request
        delay = random.uniform(self.min_delay, self.max_delay)
        if self.debug:
            spider.logger.info(f'LinkedInMiddleware: Sleeping for {delay} seconds before LinkedIn request')
        time.sleep(delay)
        
        return None
        
    def process_response(self, request, response, spider):
        # Only apply special handling to LinkedIn URLs
        if 'linkedin.com' not in request.url:
            return response
            
        # Handle 999 status code (LinkedIn's anti-bot detection)
        if response.status == 999:
            spider.logger.warning(f"LinkedIn anti-bot detection triggered! URL: {request.url}")
            # Significantly increase wait time when detected
            time.sleep(random.uniform(60, 120))
            reason = "linkedin_antibot_triggered"
            return self._retry(request, reason, spider) or response
            
        # Handle 429 (Too Many Requests)
        if response.status == 429:
            spider.logger.warning(f"Rate limit hit (429) for: {request.url}")
            # Wait longer when rate limited
            time.sleep(random.uniform(30, 60))
            reason = response_status_message(response.status)
            return self._retry(request, reason, spider) or response
        
        return response


class DDGItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    text = scrapy.Field()
    html = scrapy.Field()
    search_query = scrapy.Field()


class DDGCrawlSpider(CrawlSpider):
    name = "ddg_crawl_spider"
    # custom_settings = {
    #     'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    #     'ROBOTSTXT_OBEY': False,
    #     'DOWNLOAD_DELAY': 1,
    #     'CONCURRENT_REQUESTS': 8,
    #     # LinkedIn specific settings
    #     'LINKEDIN_MIN_DELAY': 3,
    #     'LINKEDIN_MAX_DELAY': 7,
    #     'LINKEDIN_DEBUG': True,
    #     # Configure cookie handling
    #     'COOKIES_ENABLED': True,
    #     # Configure retry settings
    #     'RETRY_ENABLED': True,
    #     'RETRY_TIMES': 3,
    #     'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429, 999],  # Added 999 for LinkedIn
    #     # Add our LinkedIn middleware
    #     'DOWNLOADER_MIDDLEWARES': {
    #         'scrapy.downloadermiddlewares.retry.RetryMiddleware': None,  # Disable default retry middleware
    #         '__main__.LinkedInMiddleware': 550,  # Add our custom middleware
    #     }
    # }

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
            )
                        ), 
                           callback='parse_page', follow=False),
    )
    
    def __init__(self, query=None, max_results=10, *args, **kwargs):
        super(DDGCrawlSpider, self).__init__(*args, **kwargs)
        self.query = query
        self.max_results = max_results
        self.ddgs = DDGS()
        self.results = []
        self.allowed_domains = []
        self.processed_urls = set()
        self.start_urls = self.get_search_results()

    def parse_start_url(self, response):
        print(f"From parse start url url is {response.url}")
        return self.parse_page(response)
    
    def get_search_results(self):
        print(f"Searching for query: {self.query} and to get {self.max_results} results")
        search_results = self.ddgs.text(self.query, max_results=self.max_results)
        urls = [result['href'] for result in search_results]
        # self.allowed_domains = list(set([self.extract_domain(url) for url in urls]))
        print(f"Allowed domains: {self.allowed_domains}")
        print(f"Allowed urls: {urls}")
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
        item['text'] = self.extract_text(response)
        item['html'] = response.text
        item['search_query'] = self.query
        
        self.results.append(dict(item))
        yield item
    
    def extract_text(self, response):
        text = ' '.join(response.xpath('//body//text()').extract())
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def run_ddg_crawl_spider(query, max_results=10, output_file=None):
    results = []
    process_settings = {'LOG_LEVEL': 'INFO'}
    if output_file:
        process_settings['FEEDS'] = {output_file: {'format': 'json', 'overwrite': True}}
    process = CrawlerProcess(process_settings)
    spider_results = []
    
    def collect_item(item, response, spider):
        spider_results.append(dict(item))
    
    dispatcher.connect(collect_item, signal=signals.item_scraped)
    process.crawl(DDGCrawlSpider, query=query, max_results=max_results)
    process.start()
    return spider_results


class EnhancedScraper:
    def __init__(self):
        self.ddgs = DDGS()
    
    def search(self, query, max_results=10):
        return self.ddgs.text(query, max_results=max_results)
    
    def scrape(self, query, max_results=10, output_file=None):
        results = run_ddg_crawl_spider(query, max_results, output_file)
        return {
            'urls': [r['url'] for r in results],
            'titles': [r['title'] for r in results],
            'extracted_texts': [r['text'] for r in results],
            'html_content': [r['html'] for r in results],
            'full_results': results
        }


if __name__ == "__main__":
    scraper = EnhancedScraper()
    results = scraper.scrape("Jupudi Industries", max_results=10, output_file="crawl_results_linkedin.json")
    print(f"Scraped {len(results['urls'])} URLs:")
    for i, url in enumerate(results['urls']):
        print(f"{i+1}. {url} - {results['titles'][i]}")
        print(f"   Text length: {len(results['extracted_texts'][i])} characters")
        print("-" * 50)