import numpy as np
import bs4
from bs4 import BeautifulSoup
from typing import List
from sentence_transformers import SentenceTransformer, util
import requests
from urllib.parse import urlparse
from tldextract import extract
import re

class URLValidator:
    """
    """
    
    def __init__(self, serpapi_key: str = ''):
        # Load SerpAPI key from instantiation
        self.serpapi_key = serpapi_key
        
        if len(self.serpapi_key) == 0:
            self.serpapi_key = None
            print("No SerpAPI key provided, citation score analysis will not be available.")
            
        self.similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.trusted_domains = None
        self.star_contributions = None
        self.link_contributions = None
        self.scores = {}
    
    def _fetch_page_soup(self) -> bs4.BeautifulSoup:
        """
        Scrapes the content of a given URL and returns a BeautifulSoup object for further processing.

        Args:
            url (str): The URL of the webpage to scrape.

        Returns:
            bs4.BeautifulSoup: The scraped content of the webpage for further processing.
        """
        try:        
            # Send a GET request to the URL with headers
            response = requests.get(self.url, timeout = 10)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            return soup

        except requests.exceptions.RequestException as e:
            return e
        
    def _extract_page_text(self) -> List[str]:
        """

        """
        # Extract text content from the webpage
        html_tags = ['p', 'span', '#text']
        text_content = []

        for tag in html_tags:
            content = [r.text.split(' ') for r in self.soup.select(tag, class_ = None)]
            content = [' '.join([string.encode("ascii", errors = "ignore").decode().strip()
                                 for string in r if len(string) > 0][:512])
                                for r in content]
            content = [r for r in content if len(r.split(' ')) > self.min_length]
            text_content += content
            
        self.page_text = list(set(text_content))

        return
    
    def _extract_outgoing_links(self) -> List[str]:
        """

        """
        base_url = urlparse(self.url).netloc
        raw_hrefs = [s.attrs['href'] for s in self.soup.find_all('a') if 'href' in s.attrs]
        outgoing_links = [urlparse(href).netloc for href in raw_hrefs]
        outgoing_links = list(set([link for link in outgoing_links if len(link) != 0
                                   and extract(link).domain != extract(self.url).domain]))
        
        self.outgoing_links = outgoing_links

        return
    
    def _get_domain_trust(self):
        self.scores['domain_trust'] = 0
        return
    
    def _get_title_relevance(self):
        query_kw = ' '.join([token for token in self.query.lower().split(' ') if len(token) > 3])
        cleaned_url = re.sub('[.!?,\'@#$%^&*()\-]', ' ', self.url).lower()
        
        title_relevance = util.pytorch_cos_sim(self.similarity_model.encode(query_kw),
                                               self.similarity_model.encode(cleaned_url)).item()
        
        self.scores['title_relevance'] = np.round(np.clip(title_relevance * 5, 1.0, 5.0), 2)
        
        return
        
    def _calculate_content_relevance(self) -> float:
        if len(self.page_text) == 0:
            self.content_relevance = 0
            return
        else:
            similarities = np.array([util.pytorch_cos_sim(self.similarity_model.encode(self.query),
                                                          self.similarity_model.encode(p)).item()
                                   for p in self.page_text])
            similarity_score = np.max(similarities) * 100
            
        if similarity_score < 30:
            similarity_score = 1.00
        elif similarity_score >= 30 and similarity_score < 50:
            similarity_score = np.round(np.clip((1 / 6) * (similarity_score - 30) + 1, a_min = 1.00, a_max = 3.00), 2)
        elif similarity_score >= 50 and similarity_score <= 70:
            similarity_score = np.round(np.clip(0.1 * ((similarity_score - 50)) + 3.0, 3.0, 5.00), 2)
        else:
            similarity_score = 5.00
            
        self.scores['content_relevance'] = similarity_score
        
        return
    
    def _check_google_scholar(self) -> int:
        """ Checks Google Scholar citations using SerpAPI. """
        params = {"q": self.url, "engine": "google_scholar", "api_key": self.serpapi_key}
        try:
            response = requests.get("https://serpapi.com/search", params = params)
            data = response.json()
            self.scores['citation_score'] = min(len(data.get("organic_results", [])) * 10, 100)  # Normalize
            return
        except:
            self.scores['citation_score'] = 0  # Default to no citations
            return
        
    def _generate_explanation(self):
        pass
    
    def _calculate_star_rating(self):
        pass
    
    def rate_url_validity(self, user_query: str, url: str, flags: dict = {'citation': False, 'domain': None}) -> dict:
        """
        Evaluates the validity of a given URL by computing various metrics including
        domain trust, content relevance, fact-checking, bias, and citation scores.

        Args:
            user_query (str): The user's original query.
            url (str): The URL to analyze.
            serp_key (str): SerpAPI key to use for additional searches.

        Returns:
            dict: A dictionary containing scores for different validity aspects.
        """
        self.query = user_query
        self.url = url
        self.min_length = len(self.url.split(' ')) * 3
        self.flags = flags

        self.soup = self._fetch_page_soup()   

        if type(self.soup) != bs4.BeautifulSoup:
            print(f"Failed to fetch content from {self.url}: {str(self.soup)}")
            return
        else:
            self._extract_page_text()
            self._extract_outgoing_links()
            self._get_domain_trust()
            self._calculate_content_relevance()
            self._get_title_relevance()

        if 'citation' in flags:
            if not self.serpapi_key:
                print("No SerpAPI key provided, citation score will not be evaluated.")
            else:
                self._check_google_scholar()
        
        self.scores['final_score'] = np.round(np.sum(np.array([score for
                                                               key, score in self.scores.items()]))\
                                              / len(self.scores), 2)
        stars, icon = 0, ''
        explanation = ''

        return {'raw_scores': self.scores, 'stars':{}, 'explanation': ''}
