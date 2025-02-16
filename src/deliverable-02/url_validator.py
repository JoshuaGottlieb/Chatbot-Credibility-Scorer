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
    
    def __init__(self, serpapi_key: str = '') -> None:
        # Load SerpAPI key from instantiation
        self._serpapi_key = serpapi_key
        
        if len(self._serpapi_key) == 0:
            self._serpapi_key = None
            print("No SerpAPI key provided, citation score analysis will not be available.")
            
        self._similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        try:
            self._trusted_domains = pd.read_csv('trusted_domains.csv')
        except:
            print("No trusted domains file provided, domain trust analysis will not be available.")    
        try:
            self._star_contributions = pd.read_csv('contribution_totals.csv').iloc[:, 1].to_numpy()
            self._link_contributions = pd.read_csv('link_scores.csv').iloc[:, 1:].to_numpy()
        except:
            print("Lookup tables for domain trust not found, defaulting to manual calculations.")
            self._star_contributions = None
            self._link_contributions = None
            
        self._reset_params()
        
        return
    
    def _reset_params(self) -> None:
        self._scores = {}
        self._soup = None
        self._page_text = None
        self._outgoing_links = None
        self._min_length = 0
        self._url = ''
        self._query = ''
        self._flags = {'citation': False, 'domain': None}
        self._stars = 0
        self._star_icon = ''
        self._explanation = ''
        return
    
    def _fetch_page_soup(self) -> None:
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
            response = requests.get(self._url, headers = headers, timeout = 10)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

            # Parse the HTML content using BeautifulSoup
            self._soup = BeautifulSoup(response.text, 'html.parser')
            return

        except requests.exceptions.RequestException as e:
            self._soup = e
            return
        
    def _extract_page_text(self) -> None:
        """

        """
        # Extract text content from the webpage
        html_tags = ['p', 'span', '#text']
        text_content = set()

        for tag in html_tags:
            for t in self._soup.select(tag, class_ = None):
                text_block = []
                for string in t.text.split(' '):
                    
                    string_cleaned = re.sub(r' ?<.+> ?', '', string.encode("ascii", errors = "ignore").decode().strip())
                    string_cleaned = re.sub(r'[\.!?,\'@#$%^&*()\-\/\:]', '', string_cleaned).lower()
#                     print(string_cleaned[:20])
                    if len(string_cleaned) > 0:
                        text_block.append(string_cleaned[:512])
#                 text_block = [re.sub(r' ?<.+> ?', '', string.encode("ascii", errors = "ignore").decode().strip())
#                               for string in t if len(string) > 0][:512]
                if len(text_block) >= self._min_length:
#                         print(text_block[0:5])
                        text_content.add(' '.join(text_block))
#             content = [r for r in content if len(r.split(' ')) > self._min_length]
#             text_content += content
            
        self._page_text = list(text_content)

        return
    
    def _extract_outgoing_links(self) -> None:
        """

        """
        base_url = urlparse(self._url).netloc
        outgoing_links = set()
        
        for a_tag in self._soup.find_all('a'):
            if 'href' in a_tag.attrs:
                link = urlparse(a_tag['href']).netloc
                if len(link) != 0 and extract(link).domain != extract(self._url).domain:
                    outgoing_links.add(link)
        
#         raw_hrefs = [s.attrs['href'] for s in self._soup.find_all('a') if 'href' in s.attrs]
#         outgoing_links = [urlparse(href).netloc for href in raw_hrefs]
#         outgoing_links = list(set([link for link in outgoing_links if len(link) != 0
#                                    and extract(link).domain != extract(self._url).domain]))
        
        self._outgoing_links = list(outgoing_links)

        return
    
    def _lookup_domain_rating(self, domain: str) -> float:
        if not self._flags['domain']:
            valid_domains = self._trusted_domains.knowledge_domain.unique()
        else:
            valid_domains = [self._flags['domain'].strip().title()]
        res = self._trusted_domains.loc[(self._trusted_domains.url == domain)
                                       & (self._trusted_domains.knowledge_domain.isin(valid_domains))]
        if len(res.index) != 0:
            return res.iloc[0].star_rating
        return 0
    
    def _get_link_contribution(self, star_rating: float, link_num: int) -> float:
        if star_rating <= 2.5:
            return 0
        star_idx = int((star_rating - 2.51) * 100)
        
        if self._link_contributions is not None:
            base_contribution = 500 * np.arctanh((star_rating - 2.5) / 4)
            return np.round(base_contribution ** (1 - ((2*(link_num - 1)) / 21)), 3)
        
        return np.round(self._link_contributions[star_idx, link_num - 1], 3)
    
    def _get_domain_trust(self) -> None:
        innate_star_rating = self._lookup_domain_rating(urlparse(self._url).netloc)
        
        if innate_star_rating == 5 or not self._outgoing_links or len(self._outgoing_links) == 0:
            self._scores['domain_trust'] = innate_star_rating
            return
        
        outgoing_domain_ratings = sorted([self._lookup_domain_rating(link) for link in self._outgoing_links],
                                         reverse = True)
        
        outgoing_contributions = np.array([self._get_link_contribution(rating, idx + 1) for
                                           idx, rating in enumerate(outgoing_domain_ratings[:12])])
        total_contributions = self._get_link_contribution(innate_star_rating + 1.5, 1) + np.sum(outgoing_contributions)
        
        if self._star_contributions is None:
            self._scores['domain_trust'] = np.round(4 * np.tanh((total_contributions) / 500) + 1, 2)
            
            return
        
        
        self._scores['domain_trust'] = (np.searchsorted(self._star_contributions,
                                                       total_contributions,
                                                       side = 'right') / 100) + 0.99
        return
    
    def _get_title_relevance(self) -> None:
        query_kw = set([token for token in self._query.lower().split(' ')])
        cleaned_url = set(re.sub(r'[\.!?,\'@#$%^&*()\-\/]', ' ', self._url).lower().split(' '))
        
        counts = np.sum(np.array([int(kw in cleaned_url) for kw in query_kw]))
        
        self._scores['title_relevance'] = np.round(np.clip((counts / len(query_kw)) * 5, 1.0, 5.0), 2)
        
        return
        
    def _calculate_content_relevance(self) -> None:
        if len(self._page_text) == 0:
            self._scores['content_relevance'] = 0
            return
        else:
            similarities = np.array([util.pytorch_cos_sim(self._similarity_model.encode(self._query),
                                                          self._similarity_model.encode(p)).item()
                                   for p in self._page_text])
            similarity_score = np.max(similarities) * 100
            
        if similarity_score < 30:
            similarity_score = 1.00
        elif similarity_score >= 30 and similarity_score < 50:
            similarity_score = np.round(np.clip((1 / 6) * (similarity_score - 30) + 1, a_min = 1.00, a_max = 3.00), 2)
        elif similarity_score >= 50 and similarity_score <= 70:
            similarity_score = np.round(np.clip(0.1 * ((similarity_score - 50)) + 3.0, 3.0, 5.00), 2)
        else:
            similarity_score = 5.00
            
        self._scores['content_relevance'] = similarity_score
        
        return
    
    def _check_google_scholar(self) -> None:
        """ Checks Google Scholar citations using SerpAPI. """
        params = {"q": self._url, "engine": "google_scholar", "api_key": self._serpapi_key}
        try:
            response = requests.get("https://serpapi.com/search", params = params)
            data = response.json()
            self._scores['citation_score'] = np.clip(len(data.get("organic_results", [])) * 0.5, 1.0, 5.0)  # Normalize
            return
        except:
            self._scores['citation_score'] = 1  # Default to no citations
            return
    
    def _determine_category_weights(self) -> dict:
        categories = list(self._scores.keys())
        
        if len(categories) == 4:
            return {'domain_trust': 0.44, 'content_relevance': 0.33, 'title_relevance': 0.01, 'citation_score': 0.24}
        
        if len(categories) == 2:
            return {'domain_trust': 0.9, 'title_relevance': 0.1}
        
        if len(categories) == 3:
            if 'content_relevance' in categories:
                return {'domain_trust': 0.55, 'content_relevance': 0.44, 'title_relevance': 0.01}
            else:
                return {'domain_trust': 0.7, 'title_relevance': 0.01, 'citation_score': 0.29}
    
    def _calculate_star_rating(self) -> None:
        weights = self._determine_category_weights()
        self._scores['final_score'] = np.round(np.sum(np.array([score * weights[key]
                                                               for key, score in self._scores.items()])), 2)
        self._stars = self._scores['final_score']
        
        full_star = "⭐"
        half_star = "🌟"
        empty_star = "☆"
        
        num_full = int(self._stars)
        if (self._stars - num_full) * 4 >= 3:
            num_full += 1
            num_half = 0
        elif (self._stars - num_full) * 4 >= 1:
            num_half = 1
        else:
            num_half = 0
        num_empty = 5 - num_full - num_half
        
        self._star_icon = (full_star * num_full) + (half_star * num_half) + (empty_star * num_empty)
        return
    
    def _generate_explanation(self) -> None:
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
                          
        for category, score in self._scores.items():
            if score >= 4:
                reasons.append(reason_options[category][0])
            elif score >= 2.5:
                reasons.append(reason_options[category][1])
            else:
                reasons.append(reason_options[category][2])
                
        self._explanation = ' '.join(reasons)
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
        
        self._query = re.sub(r'[\.!?,\'@#$%^&*()\-\/]', '', user_query).lower()
        self._url = url
        self._min_length = len(self._url.split(' ')) * 3
        self._flags['citation'] = False if 'citation' not in flags else flags['citation']
        self._flags['domain'] = None if 'domain' not in flags else flags['domain']

        self._fetch_page_soup()   

        if type(self._soup) != bs4.BeautifulSoup:
            print(f"Failed to fetch content from {self._url}: {str(self._soup)}")
            print(f"Content relevance score and outgoing link credibility cannot be calculated.")
        else:
            self._extract_page_text()
            self._extract_outgoing_links()
            self._calculate_content_relevance()
            
        self._get_domain_trust()
        self._get_title_relevance()

        if flags['citation']:
            if not self._serpapi_key:
                print("No SerpAPI key provided, citation score will not be evaluated.")
            else:
                self._check_google_scholar()
        
        self._generate_explanation()
        self._calculate_star_rating()

        return {
            'raw_scores': self._scores,
            'stars': {
                'score': self._stars,
                'icon': self._star_icon
            },
            'explanation': self._explanation
        }