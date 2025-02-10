import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

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
    page_text = scrape_url_content(url)
    
    if type(page_text) != str:
        return {"error": f"Failed to fetch content: {str(page_text)}"}

    # === Step 2: Domain Authority Check (Moz API) ===
    # Replace with actual Moz API call
    domain_trust = 60  # Placeholder value (Scale: 0-100)

    # === Step 3: Content Relevance (Semantic Similarity using Hugging Face) ===
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    similarity_score = util.pytorch_cos_sim(model.encode(user_query), model.encode(page_text)).item() * 100

    # === Step 4: Fact-Checking (Google Fact Check API) ===
    fact_check_score = check_facts(page_text)

    # === Step 5: Bias Detection (NLP Sentiment Analysis) ===
    sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
    sentiment_result = sentiment_pipeline(page_text[:512])[0]  # Process first 512 characters
    bias_score = 100 if sentiment_result["label"] == "POSITIVE" else 50 if sentiment_result["label"] == "NEUTRAL" else 30

    # === Step 6: Citation Check (Google Scholar via SerpAPI) ===
    citation_count = check_google_scholar(url, serp_key)
    citation_score = min(citation_count * 10, 100)  # Normalize

    # === Step 7: Compute Final Validity Score ===
    final_score = (
        (0.3 * domain_trust) +
        (0.3 * similarity_score) +
        (0.2 * fact_check_score) +
        (0.1 * bias_score) +
        (0.1 * citation_score)
    )

    return {
        "Domain Trust": domain_trust,
        "Content Relevance": similarity_score,
        "Fact-Check Score": fact_check_score,
        "Bias Score": bias_score,
        "Citation Score": citation_score,
        "Final Validity Score": final_score
    }

# === Helper Function: Scrape Page Content using BeautifulSoup ===
def scrape_url_content(url: str) -> str:
    """
    Scrapes the content of a given URL and returns it as a single string.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        str: The textual content of the webpage.
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

        # Extract text content from the webpage
        content = " ".join([p.text for p in soup.find_all("p")])  # Extract paragraph text

        return content

    except requests.exceptions.RequestException as e:
        return e

# === Helper Function: Fact-Checking via Google API ===
def check_facts(text: str) -> int:
    """
    Cross-checks text against Google Fact Check API.
    Returns a score between 0-100 indicating factual reliability.
    """
    api_url = f"https://toolbox.google.com/factcheck/api/v1/claimsearch?query={text[:200]}"
    try:
        response = requests.get(api_url)
        data = response.json()
        if "claims" in data and data["claims"]:
            return 80  # If found in fact-checking database
        return 40  # No verification found
    except:
        return 50  # Default uncertainty score


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