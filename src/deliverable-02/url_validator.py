import numpy as np
import pandas as pd
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
        try:
            self.trusted_domains = pd.read_csv('trusted_domains.csv')
        except:
            print("No trusted domains file provided, domain trust analysis will not be available.")    
        try:
            self.star_contributions = pd.read_csv('contribution_totals.csv').iloc[:, 1].to_numpy()
            self.link_contributions = pd.read_csv('link_scores.csv').iloc[:, 1:].to_numpy()
        except:
            print("Lookup tables for domain trust not found, defaulting to manual calculations.")
            self.star_contributions = None
            self.link_contributions = None
            
        self._reset_params()
        
        return
    
    def _reset_params(self):
        self.scores = {}
        self.soup = None
        self.page_text = None
        self.outgoing_links = None
        self.min_length = 0
        self.url = ''
        self.query = ''
        self.flags = {'citation': False, 'domain': None}
        self.stars = 0
        self.star_icon = ''
        self.explanation = ''
        return
    
    def _fetch_page_soup(self) -> bs4.BeautifulSoup:
        """
        Scrapes the content of a given URL and returns a BeautifulSoup object for further processing.

        Args:
            url (str): The URL of the webpage to scrape.

        Returns:
            bs4.BeautifulSoup: The scraped content of the webpage for further processing.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        try:        
            # Send a GET request to the URL with headers
            response = requests.get(self.url, headers = headers, timeout = 10)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

            # Parse the HTML content using BeautifulSoup
            self.soup = BeautifulSoup(response.text, 'html.parser')
            return

        except requests.exceptions.RequestException as e:
            self.soup = e
            return
        
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
    
    def _lookup_domain_rating(self, domain: str) -> float:
        if not self.flags['domain']:
            valid_domains = self.trusted_domains.knowledge_domain.unique()
        else:
            valid_domains = [self.flags['domain'].strip().title()]
        res = self.trusted_domains.loc[(self.trusted_domains.url == domain)
                                       & (self.trusted_domains.knowledge_domain.isin(valid_domains))]
        if len(res.index) != 0:
            return res.iloc[0].star_rating
        return 0
    
    def _get_link_contribution(self, star_rating: float, link_num: int) -> float:
        if star_rating <= 2.5:
            return 0
        star_idx = int((star_rating - 2.51) * 100)
        
        if self.link_contributions is not None:
            base_contribution = 500 * np.arctanh((star_rating - 2.5) / 4)
            return np.round(base_contribution ** (1 - ((2*(link_num - 1)) / 21)), 3)
        
        return np.round(self.link_contributions[star_idx, link_num - 1], 3)
    
    def _get_domain_trust(self) -> float:
        innate_star_rating = self._lookup_domain_rating(urlparse(self.url).netloc)
        
        if innate_star_rating == 5 or not self.outgoing_links or len(self.outgoing_links) == 0:
            self.scores['domain_trust'] = innate_star_rating
            return
        
        outgoing_domain_ratings = sorted([self._lookup_domain_rating(link) for link in self.outgoing_links],
                                         reverse = True)
        
        outgoing_contributions = np.array([self._get_link_contribution(rating, idx + 1) for
                                           idx, rating in enumerate(outgoing_domain_ratings[:12])])
        total_contributions = self._get_link_contribution(innate_star_rating + 1.5, 1) + np.sum(outgoing_contributions)
        
        if self.star_contributions is None:
            self.scores['domain_trust'] = np.round(4 * np.tanh((total_contributions) / 500) + 1, 2)
            
            return
        
        
        self.scores['domain_trust'] = (np.searchsorted(self.star_contributions,
                                                       total_contributions,
                                                       side = 'right') / 100) + 0.99
        return
    
    def _get_title_relevance(self) -> float:
        query_kw = [token for token in self.query.lower().split(' ') if len(token) > 3]
        cleaned_url = re.sub(r'[\.!?,\'@#$%^&*()\-\/]', ' ', self.url).lower().split(' ')
        
        counts = np.sum(np.array([1 if cleaned_url.count(kw) > 0 else 0 for kw in query_kw]))
        
        self.scores['title_relevance'] = np.round(np.clip((counts / len(query_kw)) * 5, 1.0, 5.0), 2)
        
        return
        
    def _calculate_content_relevance(self) -> float:
        if len(self.page_text) == 0:
            self.scores['content_relevance'] = 0
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
    
    def _check_google_scholar(self) -> float:
        """ Checks Google Scholar citations using SerpAPI. """
        params = {"q": self.url, "engine": "google_scholar", "api_key": self.serpapi_key}
        try:
            response = requests.get("https://serpapi.com/search", params = params)
            data = response.json()
            self.scores['citation_score'] = np.clip(len(data.get("organic_results", [])) * 0.5, 1.0, 5.0)  # Normalize
            return
        except:
            self.scores['citation_score'] = 1  # Default to no citations
            return
    
    def _determine_category_weights(self) -> dict:
        categories = list(self.scores.keys())
        
        if len(categories) == 4:
            return {'domain_trust': 0.44, 'content_relevance': 0.33, 'title_relevance': 0.01, 'citation_score': 0.24}
        
        if len(categories) == 2:
            return {'domain_trust': 0.9, 'title_relevance': 0.1}
        
        if len(categories) == 3:
            if 'content_relevance' in categories:
                return {'domain_trust': 0.55, 'content_relevance': 0.44, 'title_relevance': 0.01}
            else:
                return {'domain_trust': 0.7, 'title_relevance': 0.01, 'citation_score': 0.29}
    
    def _calculate_star_rating(self):
        weights = self._determine_category_weights()
        self.scores['final_score'] = np.round(np.sum(np.array([score * weights[key]
                                                               for key, score in self.scores.items()])), 2)
        self.stars = self.scores['final_score']
        
        full_star = "⭐"
        half_star = "🌟"
        empty_star = "☆"
        
        num_full = int(self.stars)
        if (self.stars - num_full) * 4 >= 3:
            num_full += 1
            num_half = 0
        elif (self.stars - num_full) * 4 >= 1:
            num_half = 1
        else:
            num_half = 0
        num_empty = 5 - num_full - num_half
        
        self.star_icon = (full_star * num_full) + (half_star * num_half) + (empty_star * num_empty)
        return
    
    def _generate_explanation(self):
        reasons = []
        
        reason_options = {
            'domain_trust':
              [
                  'The webpage domain is highly trusted, or the webpage links to many credible domains.',
                  'The webpage domain is moderately trusted, or the webpage contains some links to credible domains.',
                  'Neither the webpage domain nor the linked domains are highly trusted.'
              ],
            'content_relevance':
            [
                'The webpage content is highly relevant to the query.',
                'The webpage content is loosely related to the query.',
                'The webpage content cannot be found or is not related to the query.'
            ],
            'title_relevance':
            [
                'The webpage url is highly relevant to the query.',
                'The webpage url is loosely related to the query.',
                'The webpage url is not related to the query.'
            ],
            'citation_score':
            [
                'The webpage is often cited by academic sources.',
                'The webpage is rarely cited by academic sources.',
                'The webpage is not cited by academic sources.'
            ]
        }
                          
        for category, score in self.scores.items():
            if score >= 4:
                reasons.append(reason_options[category][0])
            elif score >= 2.5:
                reasons.append(reason_options[category][1])
            else:
                reasons.append(reason_options[category][2])
                
        self.explanation = ' '.join(reasons)
        return
    
    def rate_url_validity(self, user_query: str, url: str,
                          flags: dict = {'citation': False, 'domain': None}) -> dict:
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
        self._reset_params()
        
        self.query = user_query
        self.url = url
        self.min_length = len(self.url.split(' ')) * 3
        self.flags['citation'] = False if 'citation' not in flags else flags['citation']
        self.flags['domain'] = None if 'domain' not in flags else flags['domain']

        self._fetch_page_soup()   

        if type(self.soup) != bs4.BeautifulSoup:
            print(f"Failed to fetch content from {self.url}: {str(self.soup)}")
            print(f"Content relevance score and outgoing link credibility cannot be calculated.")
        else:
            self._extract_page_text()
            self._extract_outgoing_links()
            self._calculate_content_relevance()
            
        self._get_domain_trust()
        self._get_title_relevance()

        if flags['citation']:
            if not self.serpapi_key:
                print("No SerpAPI key provided, citation score will not be evaluated.")
            else:
                self._check_google_scholar()
        
        self._generate_explanation()
        self._calculate_star_rating()

        return {
            'raw_scores': self.scores,
            'stars': {
                'score': self.stars,
                'icon': self.star_icon
            },
            'explanation': self.explanation
        }