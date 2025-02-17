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
    A URL validation class that evaluates the credibility of a webpage using multiple factors:
        domain trust: A rating of the innate credibility of the domain and outgoing sources cited in the page
        content relevance: A rating of the semantic similarity between a user query and the page content
        title relevance: A rating of the presence of keywords in the user query contained in the webpage URL
        citation score: A rating of the appearance of the URL in a Google Scholar search
    """
    
    def __init__(self, serpapi_key: str = '') -> None:
        # Load SerpAPI key from instantiation
        self._serpapi_key = serpapi_key
        
        # If the SerpAPI key is not provided, print a warning.
        if len(self._serpapi_key) == 0:
            self._serpapi_key = None
            print('No SerpAPI key provided, citation score analysis will not be available.')
            
        # Load models and lookup tables from API calls and local .csv files
        self._similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        try:
            self._trusted_domains = pd.read_csv('./tables/trusted_domains.csv')
        except:
            print('No trusted domains file provided, domain trust analysis will not be available.')    
        try:
            self._star_contributions = pd.read_csv('./tables/contribution_totals.csv').iloc[:, 1].to_numpy()
            self._link_contributions = pd.read_csv('./tables/link_scores.csv').iloc[:, 1:].to_numpy()
        except:
            print('Lookup tables for domain trust not found, defaulting to manual calculations.')
            self._star_contributions = None
            self._link_contributions = None
            
        # Initialize other class attributes
        self._reset_params()
        
        return
    
    def _reset_params(self) -> None:
        """ Resets class parameters to a default state to prepare for new URL validation. """
        
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
        """ Scrapes the content of a given URL and extracts a BeautifulSoup object for further processing. """
        
        # Declare headers for request to help prevent scraping attempts from being blocked
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
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
        """ Extracts textual content from a scraped webpage for use in semantic comparison. """
        
        # Extract text content from the webpage
        html_tags = ['p', 'span', '#text'] # Classes/CSS Selectors to use
        text_content = set()

        for tag in html_tags:
            for t in self._soup.select(tag, class_ = None):
                text_block = []
                for string in t.text.split(' '):
                    # Clean the string by removing any extracted HTML tags and remove non-ASCII characters
                    string_cleaned = re.sub(r' ?<.+> ?', '',
                                            string.encode('ascii', errors = 'ignore').decode().strip())
                    # Clean the string by removing many special characters
                    string_cleaned = re.sub(r'[\.!?,\'@#$%^&*()\-\/\:]', '', string_cleaned).lower()
                    if len(string_cleaned) > 0:
                        text_block.append(string_cleaned[:512]) # Extract only the first 512 characters
                # Add text block if it contains at least the minimum number of tokens
                if len(text_block) >= self._min_length:
                        text_content.add(' '.join(text_block))
            
        self._page_text = list(text_content)

        return
    
    def _extract_outgoing_links(self) -> None:
        """ Extracts all outgoing links on a scraped webpage to other domains. """
        
        outgoing_links = set()
        
        for a_tag in self._soup.find_all('a'):
            if 'href' in a_tag.attrs:
                link = urlparse(a_tag['href']).netloc # Extract only the base urls
                # If the link is empty or from the same domain as the scraped webpage, ignore the link
                if len(link) != 0 and extract(link).domain != extract(self._url).domain:
                    outgoing_links.add(link)
        
        self._outgoing_links = list(outgoing_links)

        return
    
    def _lookup_domain_rating(self, domain: str) -> float:
        """
        Looks up the star rating for a given domain from a table of trusted sources.
        
        Args:
            domain (str): The domain to evaluate.
            
        Returns:
            float: A star rating between 1.00 to 5.00 denoting the innate domain trustworthiness.
        """
        
        # Check whether to restrict search to a specific knowledge domain
        all_domains = self._trusted_domains.knowledge_domain.unique()
        
        if not self._flags['domain']:
            valid_domains = all_domains
        else:
            # If a specific knowledge domain is provided, see if it exists in the lookup, else use all domains
            flag_domain = self._flags['domain'].strip().title()
            valid_domains = [flag_domain] if flag_domain in all_domains else all_domains
            
        # Search through pre-loaded Pandas dataframe
        res = self._trusted_domains.loc[(self._trusted_domains.url == domain)
                                       & (self._trusted_domains.knowledge_domain.isin(valid_domains))]
        # If found, return star rating, else return 0
        if len(res.index) != 0:
            return res.iloc[0].star_rating
        return 0
    
    def _get_link_contribution(self, star_rating: float, link_num: int) -> float:
        """
        Looks up or calculates the contribution score for a given link.
        
        Args:
            star_rating (float): The trustworthiness of the link as a star rating from 1.00 to 5.00.
            link_num (int): The 1-based index of which link is being evaluated.
        Returns:
            float: The contribution score for the link.
        """
        
        # The formula works only for links with a star rating greater than 2.5
        if star_rating <= 2.5:
            return 0
        
        # If no lookup table is set, manually calculate the contribution score
        if self._link_contributions is not None:
            base_contribution = 500 * np.arctanh((star_rating - 2.5) / 4)
            return np.round(base_contribution ** (1 - ((2*(link_num - 1)) / 21)), 3)
        
        # Calculate the index needed for direct access to the lookup table
        star_idx = int((star_rating - 2.51) * 100)
        
        return np.round(self._link_contributions[star_idx, link_num - 1], 3)
    
    def _get_domain_trust(self) -> None:
        """
        Calculates the domain trustworthiness based on the innate rating of the domain and
        the innate rating of all domains present in outgoing links.
        """
        
        # Lookup the innate rating, if any
        innate_star_rating = self._lookup_domain_rating(urlparse(self._url).netloc)
        
        # If the domain is perfectly trustworthy (star_rating 5), we are done
        # If there are no outgoing links because the page was unable to be scraped or did
        # not contain any links, we are done
        if innate_star_rating == 5 or not self._outgoing_links or len(self._outgoing_links) == 0:
            self._scores['domain_trust'] = innate_star_rating
            return
        
        # Get star ratings and contributions for outgoing links
        outgoing_domain_ratings = sorted([self._lookup_domain_rating(link) for link in self._outgoing_links],
                                         reverse = True)
        
        outgoing_contributions = np.array([self._get_link_contribution(rating, idx + 1) for
                                           idx, rating in enumerate(outgoing_domain_ratings[:12])])
        
        # Calculate the total contribution score, including the contribution score for the webpage itself
        total_contributions = self._get_link_contribution(innate_star_rating + 1.5, 1) + np.sum(outgoing_contributions)
        
        # If a star contribution lookup table is not available, calculate the star rating manually
        if self._star_contributions is None:
            self._scores['domain_trust'] = np.round(4 * np.tanh((total_contributions) / 500) + 1, 2)
            return
        
        # If the lookup table is available, search the array for the maximal value less than
        # total_contributions and convert the found index to a star rating
        self._scores['domain_trust'] = (np.searchsorted(self._star_contributions,
                                                       total_contributions,
                                                       side = 'right') / 100) + 0.99
        return
    
    def _get_title_relevance(self) -> None:
        """ Calculates title relevance by calculating URL similarity to the user query. """
        # Extract unique words from the query
        query_kw = set([token for token in self._query.lower().split(' ')])
        
        # Clean the url by replacing special characters with spaces
        cleaned_url = set(re.sub(r'[\.!?,\'@#$%^&*()\-\/]', ' ', self._url).lower().split(' '))
        
        # Get the proportion of the keywords in the URL and normalize
        counts = np.sum(np.array([int(kw in cleaned_url) for kw in query_kw]))
        self._scores['title_relevance'] = np.round(np.clip((counts / len(query_kw)) * 5, 1.0, 5.0), 2)
        
        return
        
    def _calculate_content_relevance(self) -> None:
        """ Calculates the content relevance by comparing the text content of the webpage to the user query. """
        
        # If no page text was scraped, return minimal score.
        if len(self._page_text) == 0:
            self._scores['content_relevance'] = 1.00
            return
        else:
            # For each text block, calculate the similarity to the user query
            similarities = np.array([util.pytorch_cos_sim(self._similarity_model.encode(self._query),
                                                          self._similarity_model.encode(p)).item()
                                   for p in self._page_text])
            # Take the maximum similarity found and scale from 0-1 to 0-100
            similarity_score = np.max(similarities) * 100
            
        # Because the semantic similarity is not linear, apply some normalization based on unit testing
        # If the similarity score is low, return 1.00
        if similarity_score < 20:
            similarity_score = 1.00
        # If the similarity score is moderate, linearly scale from 1.00 to 3.00
        elif similarity_score >= 20 and similarity_score <= 50:
            similarity_score = np.round(np.clip((1 / 15) * (similarity_score - 20) + 1, a_min = 1.00, a_max = 3.00), 2)
        # If the similarity score is high, scale from 3.00 to 5.00
        elif similarity_score > 50 and similarity_score <= 75:
            similarity_score = np.round(np.clip(0.08 * ((similarity_score - 50)) + 3.0, 3.0, 5.00), 2)
        else:
            similarity_score = 5.00
            
        self._scores['content_relevance'] = similarity_score
        
        return
    
    def _check_google_scholar(self) -> None:
        """ Checks Google Scholar citations using SerpAPI. """
        
        params = {'q': self._url, 'engine': 'google_scholar', 'api_key': self._serpapi_key}
        try:
            response = requests.get('https://serpapi.com/search', params = params)
            data = response.json()
            self._scores['citation_score'] = np.clip(len(data.get('organic_results', [])) * 0.5, 1.0, 5.0)  # Normalize
            return
        except:
            self._scores['citation_score'] = 1  # Default to no citations
            return
    
    def _determine_category_weights(self) -> dict:
        """
        Determines how to weight scoring criteria based on which scoring criteria are present.
        
        Returns:
            dict: A dictionary of weights adding up to 1 for all present categories.
        """
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
        """ Calculates the final star rating (1-5) and generates a string of stars as a visualization. """
        
        # Determine and apply weights
        weights = self._determine_category_weights()
        self._scores['final_score'] = np.round(np.sum(np.array([score * weights[key]
                                                               for key, score in self._scores.items()])), 2)
        self._stars = self._scores['final_score']
        
        # Define emoji codes for full, half, and empty stars
        full_star = '⭐'
        half_star = '🌟'
        empty_star = '☆'
        
        # Round to the nearest half-star
        # Ex: 3.75 = 4 full stars, 3.24 = 3 full stars, and 3.25-3.74 = 3 full stars and 1 half star
        num_full = int(self._stars)
        if (self._stars - num_full) * 4 >= 3:
            num_full += 1
            num_half = 0
        elif (self._stars - num_full) * 4 >= 1:
            num_half = 1
        else:
            num_half = 0
        num_empty = 5 - num_full - num_half
        
        # Generate star icon string
        self._star_icon = (full_star * num_full) + (half_star * num_half) + (empty_star * num_empty)
        
        return
    
    def _generate_explanation(self) -> None:
        """ Generates a human-readable explanation for the score. """
        
        reasons = []
        
        # Define reasonings for each scoring criterion
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
        
        # Apply appropriate string based on strength for that scoring criterion
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
        Main function to evaluate the validity of a webpage.

        Args:
            user_query (str): The user's original query.
            url (str): The URL to analyze.
            flags (dict): Flags to use to enable citation scoring or to restrict
                            domain trust to a specific knowledge domain such as Finance.

        Returns:
            dict: A dictionary containing scores for different validity aspects, a human-readable star rating,
                    and a human-readable explanation for the rating of the webpage.
        """
        
        # Reset modifiable parameters to clear the instance from any previous sessions
        self._reset_params()
        
        # Extract, clean, validate, and derive all parameters from the function arguments
        self._query = re.sub(r'[\.!?,\'@#$%^&*()\-\/]', '', user_query).lower()
        self._url = url
        self._min_length = len(self._url.split(' ')) * 3
        self._flags['citation'] = False if 'citation' not in flags else flags['citation']
        self._flags['domain'] = None if 'domain' not in flags else flags['domain']

        # Attempt to scrape the webpage
        self._fetch_page_soup()   

        # If the scraping failed, notify the user and skip any steps which require webpage content
        if type(self._soup) != bs4.BeautifulSoup:
            print(f'Failed to fetch content from {self._url}: {str(self._soup)}')
            print(f'Content relevance score and outgoing link credibility cannot be calculated.')
        else:
            # If the scraping succeeded, extract the page text, outgoing links, and calculate the content relevance
            self._extract_page_text()
            self._extract_outgoing_links()
            self._calculate_content_relevance()
            
        # Calculate domain trust and title relevance
        self._get_domain_trust()
        self._get_title_relevance()

        # If the user asked for citation scoring and the instance has a SerpAPI key, calculate citation score
        if flags['citation']:
            if not self._serpapi_key:
                print('No SerpAPI key provided, citation score will not be evaluated.')
            else:
                self._check_google_scholar()
        
        # Generate human-readable explanations and final star rating
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