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
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel,Field
from langgraph.graph import StateGraph
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from urllib.parse import quote_plus 
import requests
from typing import TypedDict

class PersonName(BaseModel):
    Name:str = Field(description="Name of the person")
    Title:str = Field(description="Title of the person in the company")

class CompanyFinancials(BaseModel):
    Revenue: str = Field(description="Revenue of the company or Turnover")
    Profit: str = Field(description="Profit of the company")
    Loss: str = Field(description="Loss of the company")
    Financial_Year:str = Field(description="Financial Year of the company")

class CompanyTaxDetails(BaseModel):
    Tax_number:str = Field(description="Tax Number of the company like GSTIN,ITIN,PAN,EIN etc.")
    Tax_Type:str = Field(description="Tax Type of the company like GST,IT,PAN etc.")
    State:str = Field(description="State of the company")
    Country:str = Field(description="Country of the company")

class CompanyContactInformation(BaseModel):
    Email:str = Field(description="Email of the company")
    Phone:str = Field(description="Phone of the company")
    Address:str = Field(description="Address of the company")
    Website:str = Field(description="Website of the company")

class CommpanyAssociatedOrganizations(BaseModel):
    Organization_Name:str = Field(description="Organization Name")
    Organization_Type:str = Field(description="Organization Type")
    Relationship:str = Field(description="Relationship with the company")

class CompanyProfile(BaseModel):
    Company_Name: str = Field(description="Company Name")
    Description: str = Field(description="Description about the company")
    Key_People: List[PersonName] = Field(description="Key People in the company")
    Number_of_Employees:str = Field(description="Number of Employees in the company")
    Major_Projects_or_Products:List[str] = Field(description="Major Projects or Products of the company")
    Financial_Information:List[CompanyFinancials] = Field(description="Financial Information of the company")
    Business_Model:str = Field(description="Business Model of the company")
    Founding_Date:str = Field(description="Founding Date of the company")
    Industry_Sector:str = Field(description="Industry Sector of the company")
    Headquarters_Location:str = Field(description="Headquarters Location of the company")
    Company_Structure:str = Field(description="Company Structure")
    Company_Contact_Information:List[CompanyContactInformation ] = Field(description="Company Contact Information")
    Key_Competitors:List[str] = Field(description="Key Competitors of the company")
    Geographies_Served:List[str] = Field(description="Geographies Served by the company")
    Number_of_Branches:str = Field(description="Number of Branches of the company")
    Tax_Details:List[CompanyTaxDetails] = Field(description="Tax Details of the company")
    Company_Registration_Details:str = Field(description="Company Registration Details of the company")
    Company_Compliance_Details:str = Field(description="Company Compliance Details of the company")
    Associated_Organizations:List[CommpanyAssociatedOrganizations] = Field(description="Associated Organizations of the company")
    Sources:List[str] = Field(description="Sources of the information")




# Custom JSON encoder to handle numpy types and other non-standard types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, set):
            return list(obj)
        return super(NumpyJSONEncoder, self).default(obj)


# Helper function to convert numpy types in a nested structure
def convert_numpy_types(obj):
    """
    Convert numpy types in a nested structure to Python native types
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


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
        """Extract named entities using Hugging Face NER pipeline with improved handling of split tokens"""
        if not text:
            return []
            
        # Process text in chunks to avoid too-long sequences
        max_length = 512
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        all_entities = []
        for chunk in chunks:
            # Get raw NER results for this chunk
            raw_entities = self.ner_pipeline(chunk)
            
            # Convert numpy values to Python native types
            raw_entities = convert_numpy_types(raw_entities)
            
            # Group entities that were split by tokenization within this chunk
            chunk_grouped_entities = []
            current_entity = None
            
            for item in raw_entities:
                # Check if this is a continuation of the previous entity (B- vs I-)
                if current_entity and item['entity'].startswith('I-') and current_entity['entity'][2:] == item['entity'][2:]:
                    # This is a continuation - update the current entity
                    if 'word' in item and item['word'].startswith('##'):
                        current_entity['word'] += item['word'][2:]  # Remove the ## prefix
                    else:
                        current_entity['word'] += item['word']
                        
                    current_entity['end'] = item['end']
                    # Use max score instead of average
                    current_entity['score'] = max(current_entity['score'], item['score'])
                else:
                    # If we had a current entity, add it to our results
                    if current_entity:
                        # Use the original text span for the full word
                        current_entity['word'] = chunk[current_entity['start']:current_entity['end']]
                        chunk_grouped_entities.append(current_entity)
                    
                    # Start a new entity
                    current_entity = item.copy()
            
            # Don't forget to add the last entity
            if current_entity:
                current_entity['word'] = chunk[current_entity['start']:current_entity['end']]
                chunk_grouped_entities.append(current_entity)
            
            all_entities.extend(chunk_grouped_entities)
        
        # Group entities by word and type (across chunks)
        final_grouped_entities = {}
        for entity in all_entities:
            entity_text = entity['word']
            entity_type = entity['entity']
            
            # Remove the B- or I- prefix for consistent grouping
            entity_type_clean = entity_type[2:] if entity_type.startswith(('B-', 'I-')) else entity_type
            
            key = (entity_text, entity_type_clean)
            if key in final_grouped_entities:
                final_grouped_entities[key]['count'] += 1
                final_grouped_entities[key]['score'] = max(final_grouped_entities[key]['score'], entity['score'])
            else:
                final_grouped_entities[key] = {
                    'text': entity_text,
                    'type': entity_type_clean,  # Use the clean type without B-/I- prefix
                    'count': 1,
                    'score': entity['score']
                }
        
        return list(final_grouped_entities.values())
    
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
        
        # Convert the item to a dict with numpy values converted to Python types
        item_dict = convert_numpy_types(dict(item))
        self.results.append(item_dict)
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
        # Convert item to dict and handle numpy types
        item_dict = convert_numpy_types(dict(item))
        spider_results.append(item_dict)
    
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
        # print(f"Inside the preprocess_data method passed data is")
        text = state.text
        # print(f"Text is Read")
        
        # Break into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Filter to more substantive paragraphs (at least 100 chars)
        paragraphs = [p for p in paragraphs if len(p) >= 100]
        
        # Take top paragraphs by length if too many
        if len(paragraphs) > 10:
            paragraphs = sorted(paragraphs, key=len, reverse=True)[:10]
        
        # print(f"Before returning preprocess_data method passed data is")
        state.paragraphs = paragraphs
        # return {"text": text, "entities": state.entities, "paragraphs": paragraphs, "url": state.url, "title": state.title}
        return state
    
    def analyze_entities(self, state):
        # print(f"Inside the analyze_entities method passed data is {state}")
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
        # Convert numpy values to native Python types before JSON serialization
        entity_text = json.dumps(convert_numpy_types(entity_groups), indent=2, cls=NumpyJSONEncoder)
        # print(f"Entity groups is {entity_text}")
        
        # Use LLM to analyze the entities
        prompt = [
            SystemMessage(content= """You are an expert at business intelligence analysis and information extraction.
            Analyze the provided named entities from web content to identify key business information.
            Focus on extracting company information, key people, organizations,Contact Details, Phone Numbers,Locations,Number of Branches and other relevant business details.
            Return your findings as a structured JSON object."""),
            HumanMessage(content= f"Here are named entities extracted from web content about a company:\n{entity_text}\n\nAnalyze these entities to identify the company name, key people, organizations, locations, and other relevant business information.")
        ]
        
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
        state.entity_analysis = entity_analysis_json
        # return {**state, "entity_analysis": entity_analysis_json}
        return state
    
    def analyze_text(self, state):
        """Process the extracted text with LLM"""

        # print(f"Inside the analyze_text method passed data is {state}")
        paragraphs = state.paragraphs
        title = state.title
        url = state.url
        
        # Join paragraphs with newlines for analysis
        text_for_analysis = "\n\n".join(paragraphs)
        
        # Create prompt for text analysis
        prompt = [
            SystemMessage(content= """You are an expert at business intelligence analysis and information extraction.
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
            - Company Contact Information
            - Company Phone Numbers
            - Key competitors
            - Geographies Served
            - Number of Branches
            - Tax related information like GSTIN, PAN ,ITIN etc.
            - Copmany Registration Details like CIN, LLPIN etc.
            - Company Financial Details like Revenue, Profit, Loss etc.
            - Company Compliance Details like ROC, MCA etc.
            - Copmany Contact Details like Email, Phone, Address etc.
            - Copmany Social Media Details like Facebook, Twitter, LinkedIn etc.
            - Company Website Details like Domain, Hosting etc.
            - Copmpany Type like Private, Public, LLP, LLC, Proprietorship etc.
            Return your findings as a structured JSON object."""),
            HumanMessage(content= f"Page title: {title}\nURL: {url}\n\nHere is text extracted from web content about a company:\n{text_for_analysis}\n\nExtract key business information from this text.")
        ]
        
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
        state.text_analysis = text_analysis_json
        # return {**state, "text_analysis": text_analysis_json}
        return state
    
    def integrate_results(self, state):
        """Combine entity and text analysis results"""
        # print(f"Inside the integrate_results method passed data is {state}")
        entity_analysis = state.entity_analysis
        text_analysis = state.text_analysis
        
        # Create prompt for integration
        combined_data = {
            "entity_analysis": entity_analysis,
            "text_analysis": text_analysis,
            "title": state.title,
            "url": state.url
        }
        
        combined_data_str = json.dumps(convert_numpy_types(combined_data), indent=2, cls=NumpyJSONEncoder)
        
        prompt = [
            SystemMessage(content = """You are an expert at business intelligence analysis and information extraction.
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
            - Company Contact Information
            - Company Phone Numbers
            - Key competitors
            - Geographies Served
            - Number of Branches
            - Tax related information like GSTIN, PAN ,ITIN etc.
            - Copmany Registration Details like CIN, LLPIN etc.
            - Company Financial Details like Revenue, Profit, Loss etc.
            - Company Compliance Details like ROC, MCA etc.
            - Copmany Contact Details like Email, Phone, Address etc.
            - Copmany Social Media Details like Facebook, Twitter, LinkedIn etc.
            - Company Website Details like Domain, Hosting etc.
            - Copmpany Type like Private, Public, LLP, LLC, Proprietorship etc.
            Return your findings as a structured JSON object."""),
            HumanMessage(content= f"Here are the two analyses to integrate:\n{combined_data_str}")
        ]
        
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
        state.integrated_results = integrated_json
        # return {**state, "integrated_results": integrated_json}
        return state
    
    def format_output(self, state):
        """Format the final output"""
        # print(f"Inside the format_output method passed data is {state}")
        integrated_results = state.integrated_results
        
        # Add source information
        final_output = {
            "business_intelligence": integrated_results,
            "source": {
                "url": state.url,
                "title": state.title
            }
        }   
        state.output = final_output
        
        # return {**state, "output": final_output}
        # print(f"Before returning frok final fomrat_output method passed data is {state}")
        return state
    
    def process(self, crawl_result):
        """Process a single crawl result"""
        # Extract the inputs
        # print(f"inside the process method passed object is {crawl_result.get('url')}")
        text = crawl_result.get('text', '')
        entities = crawl_result.get('entities', [])
        url = crawl_result.get('url', '')
        title = crawl_result.get('title', '')
        
        # Convert any numpy types to Python native types
        entities = convert_numpy_types(entities)
        
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

        
        # initial_state = {}
        # initial_state['text'] = text
        # initial_state['entities'] = entities
        # initial_state['paragraphs'] = []
        # initial_state['url'] = url
        # initial_state['title'] = title
        # initial_state['entity_analysis'] = {}
        # initial_state['text_analysis'] = {}
        # initial_state['integrated_results'] = {}
        # initial_state['output'] = {}

        # print(f"initial state object is {initial_state}")
        # Run the processing chain
        try:
            result = self.chain.invoke(initial_state)
            # print(f"AFter invoking the graph type is {type(result)} is and result is {result}")
            # print(f"output is {result['output']}")
            return result.get('output', None)
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return {
                "error": str(e),
                "url": url,
                "title": title
            }


class GraphState(BaseModel):
    """State object for the LangGraph workflow"""
    text: str
    entities: List[Dict]
    paragraphs: List[str]
    url: str
    title: str
    entity_analysis: Dict
    text_analysis: Dict
    integrated_results: Dict
    output: Dict


class EnhancedScraper:
    def __init__(self, openai_api_key=None):
        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
    }
        self.ddgs = DDGS(headers=headers)
        self.urls = []
        self.openai_api_key = openai_api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize LLM processor
        self.llm_processor = LLMProcessor()
        
        # Patterns for filtering irrelevant URLs
        self.irrelevant_patterns = [
            r'wikipedia\.org',         # Often too general
            r'facebook\.com',          # Social media
            r'twitter\.com',           # Social media
            r'instagram\.com',         # Social media
            r'tiktok\.com',            # Social media
            r'youtube\.com',           # Video content
            r'glassdoor\.com',         # Employee reviews
            r'indeed\.com',            # Job listings
            r'yelp\.com',              # Reviews
            r'amazon\.com',            # Product listings
            r'bloomberg\.com',         # News articles
            r'reuters\.com',           # News articles
            r'forbes\.com',            # News articles
            r'wsj\.com',               # News articles
            r'/news/',                 # News sections
            r'/press-release/',        # Press releases
            r'/blog/',                 # Blog posts
            r'/events/',               # Events pages
        ]
    
    def search_ddg(self,query, max_results=10):
        results = []
        
        # Format the query for URL
        encoded_query = quote_plus(query)
        
        # Set up headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://duckduckgo.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # DuckDuckGo search URL
        url = f'https://html.duckduckgo.com/html/?q={encoded_query}'
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            print(f"Search response status: {response.status_code}")
            print(f"Search response : {response}")
            
            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            search_items = soup.find_all('div', class_='result__body')
            
            for item in search_items:
                if len(results) >= max_results:
                    break
                    
                title_element = item.find('a', class_='result__a')
                snippet_element = item.find('a', class_='result__snippet')
                href_element = item.find('a', class_='result__url')
                
                if title_element and href_element:
                    title = title_element.text.strip()
                    link = title_element['href']
                    
                    # Extract the actual URL from DuckDuckGo's redirect URL
                    if link.startswith('/'):
                        # Parse out the actual URL if it's a redirect
                        if 'uddg=' in link:
                            import urllib.parse
                            link = urllib.parse.unquote(link.split('uddg=')[1].split('&')[0])
                    
                    snippet = snippet_element.text.strip() if snippet_element else ""
                    
                    results.append({
                        'title': title,
                        'href': link,
                        'body': snippet
                    })
            
            return results
                
        except Exception as e:
            print(f"Error during search: {e}")
            return results

    def search(self, query, max_results=10):
        search_results = self.ddgs.text(query, max_results=max_results)  # Get more results to account for filtering
        # search_results = self.search_ddg(query, max_results=max_results)
        urls = [result['href'] for result in search_results]
        print(f"Initial search results: {len(urls)} URLs")
        
        # Filter out irrelevant URLs
        filtered_urls = self.filter_urls(urls)
        print(f"After filtering: {len(filtered_urls)} URLs")
        
        # Take only the requested number of results
        return filtered_urls[:max_results]
        
    def filter_urls(self, urls):
        """Filter out irrelevant URLs based on patterns"""
        filtered_urls = []
        for url in urls:
            # Skip if URL matches any irrelevant pattern
            if any(re.search(pattern, url, re.IGNORECASE) for pattern in self.irrelevant_patterns):
                print(f"Filtering out irrelevant URL: {url}")
                continue
            
            filtered_urls.append(url)
        return filtered_urls

    def read_json_file(self,file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

       
    def scrape(self, query, max_results=10, output_file=None):
        """Scrape content, extract entities, and process with LLM"""
        # First get search results
        self.urls = self.search(query, max_results)
        
        # Run the spider to crawl and extract content
        print(f"Crawling {len(self.urls)} URLs...")
        crawl_results = run_ddg_crawl_spider(query, max_results, self.urls, output_file)
        # file_path = '/Users/jupudi/ar_process/research_agent/company_intelligence_with_entities_jupudi.json'
        # crawl_results = self.read_json_file(file_path) 
        print(f"Extracted {len(crawl_results)} pages")
        # Filter crawl results for relevance
        relevant_results = self.filter_crawl_results(crawl_results, query)
        print(f"Filtered to {len(relevant_results)} relevant pages")
        
        # Process each result with the LLM processor
        print(f"Processing {len(relevant_results)} pages with NER and LLM...")
        processed_results = []
        for result in relevant_results:
            processed_result = self.llm_processor.process(result)
            processed_results.append(processed_result)
        
        # Save the enhanced results if output file provided
        if output_file:
            enhanced_output_file = output_file.replace('.json', '_enhanced.json')
            with open(enhanced_output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_results, f, indent=2, cls=NumpyJSONEncoder)
        
        return processed_results
        
    def filter_crawl_results(self, crawl_results, query):
        """Filter crawled pages to ensure they're relevant to the query"""
        relevant_results = []
        company_name = query.lower()

        
        for result in crawl_results:
            # print(f"Processing URL: {result.get('url')}")
            # Extract content
            title = result.get('title', '')
            if title is not None:
                title = title.lower()
            else:
                title = ''
            text = result.get('text', '')
            if text is not None:    
                text = text.lower()
            else:        
                text = ''
            url = result.get('url', '')
            if url is not None:
                url = url.lower()
            else:
                url = ''
            
            # Check if content is too short (likely not informative)
            if len(text) < 500:
                print(f"Skipping short content URL: {url}")
                continue
                
            # Skip if the URL matches irrelevant patterns
            if any(re.search(pattern, url, re.IGNORECASE) for pattern in self.irrelevant_patterns):
                print(f"Skipping irrelevant URL: {url}")
                continue
            
            # Check if company name appears in title or early in text
            # If company name doesn't appear prominently, page might not be about the company
            first_paragraph = text[:1000]
            if company_name not in title and company_name not in first_paragraph:
                mentions_count = text.count(company_name)
                # If the company is rarely mentioned in the text, skip this result
                if mentions_count < 3:
                    print(f"Skipping URL with few company mentions: {url} with mentions_count: {mentions_count}")
                    continue
            
            # Calculate a relevance score
            score = 0
            if company_name in title:
                score += 5
            score += text[:2000].count(company_name) * 2  # Early mentions weighted higher
            score += text[2000:].count(company_name)
            
            # Only include pages with a minimum relevance score
            if score >= 3:
                # Add score to the result for potential further filtering
                result['relevance_score'] = score
                relevant_results.append(result)
            else:
                print(f"Skipping low relevance URL: {url} with score: {score}")
        
        # Sort by relevance score
        relevant_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_results
    
    def generate_report(self, processed_results, output_file=None):
        """Generate a consolidated report from all processed results"""
        if not processed_results:
            return {"error": "No results to generate report"}
        
        # Convert any numpy types in the results to ensure proper JSON serialization
        processed_results = convert_numpy_types(processed_results)
        
        # Create a summary prompt with all the processed results
        results_text = json.dumps(processed_results, indent=2, cls=NumpyJSONEncoder)
        
        prompt = [
            SystemMessage(content= """You are an expert at business intelligence analysis and report generation.
            You have been given a set of structured business information extracted from multiple web pages.
            Your task is to create a comprehensive business profile by integrating all these information sources.
            Focus on consolidating information and resolving any conflicts and redundancy between sources.
            When resolving conflicts give weightage to the confidence score and select the information with higher confidence.
            When eliminating duplicates, prefer more specific and detailed information along with higher confidence scores.
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
            - Company Contact Information
            - Company Phone Numbers
            - Key competitors
            - Short summary of the business
            - Major Clients Served
            - Geographies Served
            - Tax related information like GSTIN, PAN ,ITIN,EIN,GST Number etc.
            - Copmany Registration Details like CIN, LLPIN etc.
            - Company Financial Details like Revenue, Profit, Loss etc.
            - Company Compliance Details like ROC, MCA etc.
            - Copmany Contact Details like Email, Phone, Address etc.
            - Copmany Social Media Details like Facebook, Twitter, LinkedIn etc.
            - Company Website Details like Domain, Hosting etc.
            - Copmpany Type like Private, Public, LLP, LLC, Proprietorship etc.
            - Any other relevant details you can extract like relationship between key people based on names,locations etc.
            - Exclude any mentions of platforms (e.g., Facebook, iOS, Android), supported languages, and policies (e.g., Privacy Policy, Cookie Policy)
            Return your findings as a structured JSON object with information categorized.
            Include a confidence score (0-10) for each piece of information and cite the source URL when possible."""),
            HumanMessage(content= f"Here are the structured results from multiple sources:\n{results_text}\n\nCreate a consolidated business profile.")
        ]
        
        # Use LLM to generate the consolidated report
        # CompanyProfile.json()
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        llm_structured_op = llm.with_structured_output(CompanyProfile,method="function_calling")
        report = llm_structured_op.invoke(prompt)
        report_json = report.json()
        try:
            # Try to convert the response to JSON
            report_json = json.loads(report_json)
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
            with open(report_output_file, 'w', encoding='utf-8') as f:
                json.dump(report_json, f, indent=2, cls=NumpyJSONEncoder)
        
        return report_json

if __name__ == "__main__":
    load_dotenv()
    # Set your OpenAI API key here or as an environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Initialize the enhanced scraper
    scraper = EnhancedScraper(openai_api_key=openai_api_key)
    
    # Specify the company to research
    company_name = "Naatu Restaurant"
    output_file = "company_intelligence_Naatu.json"
    
    # Run the full pipeline
    print(f"Starting research on: {company_name}")
    processed_results = scraper.scrape(company_name, max_results=10, output_file=output_file)
    # processed_results=scraper.read_json_file('/Users/jupudi/ar_process/research_agent/company_intelligence_enhanced.json')
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