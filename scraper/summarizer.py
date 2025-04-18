import scrapy
from scrapy.crawler import CrawlerProcess
from duckduckgo_search import DDGS
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.http import HtmlResponse
import json
from typing import List, Dict, Set, Any, Optional
from scrapy import signals
from scrapy.signalmanager import dispatcher
from bs4 import BeautifulSoup
import re
import os
from transformers import pipeline
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
import pandas as pd


class DDGItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    text = scrapy.Field()
    html = scrapy.Field()
    search_query = scrapy.Field()
    entities = scrapy.Field()  # New field for NER results


class DDGCrawlSpider(CrawlSpider):
    name = "ddg_crawl_spider"
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS': 8,
    }

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
    
    def __init__(self, query=None, url_list=[], max_results=10, *args, **kwargs):
        super(DDGCrawlSpider, self).__init__(*args, **kwargs)
        self.query = query
        self.max_results = max_results
        self.ddgs = DDGS()
        self.results = []
        self.allowed_domains = []
        self.processed_urls = set()
        self.start_urls = url_list
        
        # Initialize NER pipeline
        self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")

    def parse_start_url(self, response):
        print(f"From parse start url: {response.url}")
        return self.parse_page(response)
    
    def get_search_results(self):
        print(f"Searching for query: {self.query} and to get {self.max_results} results")
        search_results = self.ddgs.text(self.query, max_results=self.max_results, backend='json')
        urls = [result['href'] for result in search_results]
        print(f"Allowed urls: {urls}")
        print(f"Duck Duck Go Search size is : {len(urls)}")
        return urls
    
    def extract_domain(self, url):
        return url.split("/")[2] if "//" in url else url
    
    def extract_entities(self, text):
        """Extract named entities using Hugging Face NER pipeline"""
        if not text:
            return []
            
        # Process text in chunks to avoid too-long sequences
        max_length = 512
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        all_entities = []
        for chunk in chunks:
            entities = self.ner_pipeline(chunk)
            all_entities.extend(entities)
        
        # Group entities by word and type
        grouped_entities = {}
        for entity in all_entities:
            entity_text = entity['word']
            entity_type = entity['entity']
            
            # Strip special tokens from beginning of words
            if entity_text.startswith('##'):
                entity_text = entity_text[2:]
                
            key = (entity_text, entity_type)
            if key in grouped_entities:
                grouped_entities[key]['count'] += 1
                grouped_entities[key]['score'] = max(grouped_entities[key]['score'], entity['score'])
            else:
                grouped_entities[key] = {
                    'text': entity_text,
                    'type': entity_type,
                    'count': 1,
                    'score': entity['score']
                }
        
        return list(grouped_entities.values())
    
    def parse_page(self, response):
        if response.url in self.processed_urls:
            print(f"Already processed URL: {response.url}")
            return
            
        self.processed_urls.add(response.url)
        item = DDGItem()
        print(f"From parse page url is {response.url}")
        
        item['url'] = response.url
        item['title'] = response.xpath('//title/text()').get()
        extracted_text = self.extract_text(response.text)
        item['text'] = extracted_text
        item['html'] = response.text
        item['search_query'] = self.query
        
        # Extract entities from the text
        item['entities'] = self.extract_entities(extracted_text)
        
        self.results.append(dict(item))
        yield item
    
    def extract_text(self, response):
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


def run_ddg_crawl_spider(query, max_results=10, url_list=[], output_file=None):
    results = []
    process_settings = {'LOG_LEVEL': 'INFO'}
    if output_file:
        process_settings['FEEDS'] = {output_file: {'format': 'json', 'overwrite': True}}
    process = CrawlerProcess(process_settings)
    spider_results = []
    
    def collect_item(item, response, spider):
        spider_results.append(dict(item))
    
    dispatcher.connect(collect_item, signal=signals.item_scraped)
    process.crawl(DDGCrawlSpider, query=query, url_list=url_list, max_results=max_results)
    process.start()
    return spider_results


class LLMProcessor:
    """Process extracted text and entities using OpenAI and LangGraph"""
    
    def __init__(self, api_key=None):
        # Set API key from environment if not provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        
        # Define the processing workflow
        self.setup_graph()
    
    def setup_graph(self):
        # Define the processing graph
        self.graph = StateGraph(GraphState)
        
        # Add preprocessing node
        self.graph.add_node("preprocess", self.preprocess_data)
        
        # Add entity analysis node
        self.graph.add_node("analyze_entities", self.analyze_entities)
        
        # Add text analysis node
        self.graph.add_node("analyze_text", self.analyze_text)
        
        # Add integration node to combine results
        self.graph.add_node("integrate_results", self.integrate_results)
        
        # Add formatting node to structure final output
        self.graph.add_node("format_output", self.format_output)
        
        # Define the flow
        self.graph.set_entry_point("preprocess")
        self.graph.add_edge("preprocess", "analyze_entities")
        self.graph.add_edge("analyze_entities", "analyze_text")
        self.graph.add_edge("analyze_text", "integrate_results")
        self.graph.add_edge("integrate_results", "format_output")
        
        # Compile the graph
        self.chain = self.graph.compile()
    
    def preprocess_data(self, state):
        """Prepare the data for processing"""
        # Extract relevant text chunks (most informative paragraphs)
        text = state.text
        
        # Break into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Filter to more substantive paragraphs (at least 100 chars)
        paragraphs = [p for p in paragraphs if len(p) >= 100]
        
        # Take top paragraphs by length if too many
        if len(paragraphs) > 10:
            paragraphs = sorted(paragraphs, key=len, reverse=True)[:10]
        
        return {"text": text, "entities": state.entities, "paragraphs": paragraphs, "url": state.url, "title": state.title}
    
    def analyze_entities(self, state):
        """Process the extracted entities"""
        entities = state.entities
        
        # Group entities by type
        entity_groups = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity)
        
        # Sort each group by count and score
        for entity_type in entity_groups:
            entity_groups[entity_type] = sorted(
                entity_groups[entity_type], 
                key=lambda x: (x['count'], x['score']), 
                reverse=True
            )
        
        # Create prompt for entity analysis
        entity_text = json.dumps(entity_groups, indent=2)
        
        # Use LLM to analyze the entities
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at business intelligence analysis and information extraction.
            Analyze the provided named entities from web content to identify key business information.
            Focus on extracting company information, key people, organizations, and other relevant business details.
            Return your findings as a structured JSON object."""),
            ("user", f"Here are named entities extracted from web content about a company:\n{entity_text}\n\nAnalyze these entities to identify the company name, key people, organizations, locations, and other relevant business information.")
        ])
        
        entity_analysis = self.llm.invoke(prompt).content
        
        try:
            # Try to convert the response to JSON
            entity_analysis_json = json.loads(entity_analysis)
        except json.JSONDecodeError:
            # If not valid JSON, extract JSON portion using regex
            json_match = re.search(r'```json\n(.*?)\n```', entity_analysis, re.DOTALL)
            if json_match:
                entity_analysis = json_match.group(1)
                entity_analysis_json = json.loads(entity_analysis)
            else:
                # Create a simple structure if JSON extraction fails
                entity_analysis_json = {
                    "extracted_info": "JSON parsing failed",
                    "raw_response": entity_analysis
                }
        
        return {**state, "entity_analysis": entity_analysis_json}
    
    def analyze_text(self, state):
        """Process the extracted text with LLM"""
        paragraphs = state.paragraphs
        title = state.title
        url = state.url
        
        # Join paragraphs with newlines for analysis
        text_for_analysis = "\n\n".join(paragraphs)
        
        # Create prompt for text analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at business intelligence analysis and information extraction.
            Analyze the provided text from web content to identify key business information.
            Focus on extracting specific details about:
            - Company name and description
            - Key people (executives, founders, etc.)
            - Number of employees
            - Major projects or products
            - Financial information
            - Business model
            - Founding date
            - Industry sector
            - Headquarters location
            - Company structure
            Return your findings as a structured JSON object."""),
            ("user", f"Page title: {title}\nURL: {url}\n\nHere is text extracted from web content about a company:\n{text_for_analysis}\n\nExtract key business information from this text.")
        ])
        
        text_analysis = self.llm.invoke(prompt).content
        
        try:
            # Try to convert the response to JSON
            text_analysis_json = json.loads(text_analysis)
        except json.JSONDecodeError:
            # If not valid JSON, extract JSON portion using regex
            json_match = re.search(r'```json\n(.*?)\n```', text_analysis, re.DOTALL)
            if json_match:
                text_analysis = json_match.group(1)
                text_analysis_json = json.loads(text_analysis)
            else:
                # Create a simple structure if JSON extraction fails
                text_analysis_json = {
                    "extracted_info": "JSON parsing failed",
                    "raw_response": text_analysis
                }
        
        return {**state, "text_analysis": text_analysis_json}
    
    def integrate_results(self, state):
        """Combine entity and text analysis results"""
        entity_analysis = state.entity_analysis
        text_analysis = state.text_analysis
        
        # Create prompt for integration
        combined_data = {
            "entity_analysis": entity_analysis,
            "text_analysis": text_analysis,
            "title": state.title,
            "url": state.url
        }
        
        combined_data_str = json.dumps(combined_data, indent=2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at business intelligence analysis and information extraction.
            You have been given two sets of analysis: one based on named entity recognition and one based on text analysis.
            Your task is to integrate these results to create a comprehensive, accurate profile of the company.
            Resolve any conflicts between the two analyses, prioritizing more specific and detailed information.
            Focus on producing a single, coherent set of information about:
            - Company name and description
            - Key people (executives, founders, etc.)
            - Number of employees
            - Major projects or products
            - Financial information
            - Business model
            - Founding date
            - Industry sector
            - Headquarters location
            - Company structure
            Return your findings as a structured JSON object."""),
            ("user", f"Here are the two analyses to integrate:\n{combined_data_str}")
        ])
        
        integrated_results = self.llm.invoke(prompt).content
        
        try:
            # Try to convert the response to JSON
            integrated_json = json.loads(integrated_results)
        except json.JSONDecodeError:
            # If not valid JSON, extract JSON portion using regex
            json_match = re.search(r'```json\n(.*?)\n```', integrated_results, re.DOTALL)
            if json_match:
                integrated_results = json_match.group(1)
                integrated_json = json.loads(integrated_results)
            else:
                # Create a simple structure if JSON extraction fails
                integrated_json = {
                    "integrated_info": "JSON parsing failed",
                    "raw_response": integrated_results
                }
        
        return {**state, "integrated_results": integrated_json}
    
    def format_output(self, state):
        """Format the final output"""
        integrated_results = state.integrated_results
        
        # Add source information
        final_output = {
            "business_intelligence": integrated_results,
            "source": {
                "url": state.url,
                "title": state.title
            }
        }
        
        return {**state, "output": final_output}
    
    def process(self, crawl_result):
        """Process a single crawl result"""
        # Extract the inputs
        text = crawl_result.get('text', '')
        entities = crawl_result.get('entities', [])
        url = crawl_result.get('url', '')
        title = crawl_result.get('title', '')
        
        # Create initial state
        initial_state = GraphState(
            text=text,
            entities=entities,
            paragraphs=[],
            url=url,
            title=title,
            entity_analysis={},
            text_analysis={},
            integrated_results={},
            output={}
        )
        
        # Run the processing chain
        try:
            result = self.chain.invoke(initial_state)
            return result.output
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return {
                "error": str(e),
                "url": url,
                "title": title
            }


class GraphState:
    """State object for the LangGraph workflow"""
    
    def __init__(
        self,
        text: str = "",
        entities: List[Dict] = None,
        paragraphs: List[str] = None,
        url: str = "",
        title: str = "",
        entity_analysis: Dict = None,
        text_analysis: Dict = None,
        integrated_results: Dict = None,
        output: Dict = None
    ):
        self.text = text
        self.entities = entities or []
        self.paragraphs = paragraphs or []
        self.url = url
        self.title = title
        self.entity_analysis = entity_analysis or {}
        self.text_analysis = text_analysis or {}
        self.integrated_results = integrated_results or {}
        self.output = output or {}


class EnhancedScraper:
    def __init__(self, openai_api_key=None):
        self.ddgs = DDGS()
        self.urls = []
        self.openai_api_key = openai_api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize LLM processor
        self.llm_processor = LLMProcessor()
    
    def search(self, query, max_results=10):
        search_results = self.ddgs.text(query, max_results=max_results)
        urls = [result['href'] for result in search_results]
        print(f"Search results: {len(urls)} URLs")
        return urls
    
    def scrape(self, query, max_results=10, output_file=None):
        """Scrape content, extract entities, and process with LLM"""
        # First get search results
        self.urls = self.search(query, max_results)
        
        # Run the spider to crawl and extract content
        print(f"Crawling {len(self.urls)} URLs...")
        crawl_results = run_ddg_crawl_spider(query, max_results, self.urls, output_file)
        
        # Process each result with the LLM processor
        print(f"Processing {len(crawl_results)} pages with NER and LLM...")
        processed_results = []
        for result in crawl_results:
            processed_result = self.llm_processor.process(result)
            processed_results.append(processed_result)
        
        # Save the enhanced results if output file provided
        if output_file:
            enhanced_output_file = output_file.replace('.json', '_enhanced.json')
            with open(enhanced_output_file, 'w') as f:
                json.dump(processed_results, f, indent=2)
        
        return processed_results
    
    def generate_report(self, processed_results, output_file=None):
        """Generate a consolidated report from all processed results"""
        if not processed_results:
            return {"error": "No results to generate report"}
        
        # Create a summary prompt with all the processed results
        results_text = json.dumps(processed_results, indent=2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at business intelligence analysis and report generation.
            You have been given a set of structured business information extracted from multiple web pages.
            Your task is to create a comprehensive business profile by integrating all these information sources.
            Focus on consolidating information and resolving any conflicts between sources.
            Include information about:
            - Company name and description
            - Key people (executives, founders, etc.)
            - Number of employees
            - Major projects or products
            - Financial information
            - Business model
            - Founding date
            - Industry sector
            - Headquarters location
            - Company structure
            Return your findings as a structured JSON object with information categorized.
            Include a confidence score (0-10) for each piece of information and cite the source URL when possible."""),
            ("user", f"Here are the structured results from multiple sources:\n{results_text}\n\nCreate a consolidated business profile.")
        ])
        
        # Use LLM to generate the consolidated report
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        report = llm.invoke(prompt).content
        
        try:
            # Try to convert the response to JSON
            report_json = json.loads(report)
        except json.JSONDecodeError:
            # If not valid JSON, extract JSON portion using regex
            json_match = re.search(r'```json\n(.*?)\n```', report, re.DOTALL)
            if json_match:
                report = json_match.group(1)
                report_json = json.loads(report)
            else:
                # Create a simple structure if JSON extraction fails
                report_json = {
                    "report": "JSON parsing failed",
                    "raw_response": report
                }
        
        # Save the report if output file provided
        if output_file:
            report_output_file = output_file.replace('.json', '_report.json')
            with open(report_output_file, 'w') as f:
                json.dump(report_json, f, indent=2)
        
        return report_json


if __name__ == "__main__":
    # Set your OpenAI API key here or as an environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Initialize the enhanced scraper
    scraper = EnhancedScraper(openai_api_key=openai_api_key)
    
    # Specify the company to research
    company_name = "Jupudi Industries"
    output_file = "company_intelligence.json"
    
    # Run the full pipeline
    print(f"Starting research on: {company_name}")
    processed_results = scraper.scrape(company_name, max_results=5, output_file=output_file)
    
    # Generate consolidated report
    print("Generating consolidated report...")
    report = scraper.generate_report(processed_results, output_file=output_file)
    
    print(f"Research complete! Results saved to {output_file}, enhanced results to {output_file.replace('.json', '_enhanced.json')}, and consolidated report to {output_file.replace('.json', '_report.json')}")
    
    # Print a summary of the report
    if isinstance(report, dict) and "error" not in report:
        company_info = report.get("company_info", {})
        print("\n===== COMPANY PROFILE SUMMARY =====")
        print(f"Company: {company_info.get('name', 'Unknown')}")
        print(f"Industry: {company_info.get('industry', 'Unknown')}")
        print(f"Founded: {company_info.get('founding_date', 'Unknown')}")
        print(f"Employees: {company_info.get('employees', 'Unknown')}")
        print(f"Headquarters: {company_info.get('headquarters', 'Unknown')}")
    else:
        print("Error generating report:", report.get("error", "Unknown error"))