import requests
import string
import numpy as np
import bs4
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from typing import List
from urllib.parse import urlparse
from tldextract import extract
import re

def rate_url_validity(user_query: str, url: str, serp_key: str) -> dict:
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

    # === Step 1: Fetch Page Content ===
    soup = scrape_url_content(url)   
    
    if type(soup) != bs4.BeautifulSoup:
        return {"error": f"Failed to fetch content: {str(soup)}"}
    
    page_text = extract_page_text(soup, len(user_query) * 3)
    outgoing_links = extract_outgoing_links(soup, url)

    # === Step 2: Domain Authority Check ===
    domain_trust = np.sum(np.array([url.count(re.sub('[.!?,\'@#$%^&*()]', '', substr.strip().lower()))
                                    for substr in user_query.split()]))

    # === Step 3: Content Relevance (Semantic Similarity using Hugging Face) ===
    similarity_score = calculate_content_relevance(user_query, page_text)
    
    # === Step 6: Citation Check (Google Scholar via SerpAPI) ===
#     citation_count = check_google_scholar(url, serp_key)
#     citation_score = min(citation_count * 10, 100)  # Normalize

    # === Step 7: Compute Final Validity Score ===
    final_score = (
        (0.6 * domain_trust) +
        (0.3 * similarity_score)
#         (0.1 * citation_score)
    )

    return {
        "Domain Trust": domain_trust,
        "Content Relevance": similarity_score,
#         "Citation Score": citation_score,
        "Final Validity Score": final_score
    }

# === Helper Function: Scrape Page Content using BeautifulSoup ===
def scrape_url_content(url: str) -> bs4.BeautifulSoup:
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
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        return soup

    except requests.exceptions.RequestException as e:
        return e

# === Helper Function: Extract Page Content using BeautifulSoup ===
def extract_page_text(soup: bs4.BeautifulSoup, min_length: int) -> List[str]:
    """
    
    """
    # Extract text content from the webpage
    html_tags = ['p', 'span', '#text']
    text_content = []

    for tag in html_tags:
        content = [r.text.split(' ') for r in soup.select(tag, class_ = None)]
        content = [' '.join([string.encode("ascii", errors = "ignore").decode().strip()
                             for string in r if len(string) > 0][:512])
                            for r in content]
        content = [r for r in content if len(r.split(' ')) > min_length]
        text_content += content

    return list(set(text_content))

# === Helper Function: Extract Link Content using BeautifulSoup ===
def extract_outgoing_links(soup: bs4.BeautifulSoup, url: str) -> List[str]:
    """
    
    """
    base_url = urlparse(url).netloc
    raw_hrefs = [s.attrs['href'] for s in soup.find_all('a') if 'href' in s.attrs]
    outgoing_links = [urlparse(href).netloc for href in raw_hrefs]
    outgoing_links = list(set([link for link in outgoing_links if len(link) != 0
                               and extract(link).domain != extract(url).domain]))
    
    return outgoing_links

def calculate_domain_trust(outgoing_links: List[str], url: str) -> None:
    trusted_links = pd.read_csv('trusted_domains.csv')
    link_scores = pd.read_csv('link_scores.csv').iloc[:, 1:].to_numpy()
    contribution_totals = pd.read_csv('contribution_totals.csv').to_numpy()
    
#     links_to_bases = [

def calculate_content_relevance(user_query: str, page_text: List[str]) -> float:
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    if len(page_text) == 0:
        similarity_score = 0
    else:
        similarities = np.array([util.pytorch_cos_sim(model.encode(user_query),
                                                      model.encode(p)).item()
                               for p in page_text])
        print(np.argmax(similarities), page_text[np.argmax(similarities)], sep = '\n')
        similarity_score = np.max(similarities) * 100

    print(f"Raw similarity score: {similarity_score}")
        
    if similarity_score < 30:
        similarity_score = 1.00
    elif similarity_score >= 30 and similarity_score < 50:
        similarity_score = np.round(np.clip((1 / 6) * (similarity_score - 30) + 1, a_min = 1.00, a_max = 3.00), 2)
    elif similarity_score >= 50 and similarity_score <= 70:
        similarity_score = np.round(np.clip(0.1 * ((similarity_score - 50)) + 3.0, 3.0, 5.00), 2)
    else:
        similarity_score = 5.00
        
    return similarity_score
# === Helper Function: Citation Count via Google Scholar API ===
def check_google_scholar(api_key: str, url: str) -> int:
    """
    Checks Google Scholar citations using SerpAPI.
    
    Args:
        api_key (str): SerpAPI key to use for query.
        url (str): The URL to check.
    
    Returns:
        int: The count of citations found.
    """
    params = {"q": url, "engine": "google_scholar", "api_key": api_key}
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        return len(data.get("organic_results", []))
    except:
        return -1  # Assume no citations found