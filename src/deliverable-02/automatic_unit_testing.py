from url_validator import URLValidator
from api_keys import serpapi_key
import numpy as np
import pandas as pd
from serpapi import GoogleSearch
from typing import List, Dict, Any

def search_serpapi(query: str, api_key: str, num_results: int = 40) -> List[Dict[str, Any]]:
    """
    Submits a search query to SerpAPI and retrieves a specified number of search result links.

    Args:
        query (str): The search term or phrase to query in SerpAPI.
        api_key (str): The API key required to authenticate with SerpAPI.
        num_results (int, optional): The number of search results to return. Defaults to 40.

    Returns:
        List[str]: A list of URLs from the organic search results, up to the specified number.
        str: An error message if the query fails.

    Raises:
        Exception: Catches any errors encountered during the API request and returns an error message.
    """
    try:
        search = GoogleSearch({
            "q": query,
            "location": "Austin, Texas, United States",
            "api_key": api_key,
            "num": num_results + 10 # Gather extra links to ensure enough are returned
        })
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        return [res['link'] for res in organic_results][:num_results]
    except Exception as e:
        return f"An error occurred when retrieving results for {query}: {e}"

def process_csv(path: str, api_key: str) -> None:
    """
    Processes a unit test CSV file containing search queries, retrieves search results 
    using SerpAPI, and evaluates the credibility of the retrieved links.

    Args:
        path (str): The file path of the CSV containing search queries and their knowledge domains.
        api_key (str): The API key required to authenticate with SerpAPI.

    Returns:
        None: The function writes two CSV files:
            - 'automatic_unit_testing.csv': Contains credibility ratings for retrieved URLs.
            - 'sample.csv': Contains a subset of URLs for manual verification.
    """
    
    # Load test queries
    test_queries = pd.read_csv(path)

    # Extract domains and queries
    knowledge_domains = test_queries['domain'].to_numpy()
    queries = test_queries['query'].to_numpy()

    # Instantiate URLValidator
    validator = URLValidator(api_key)

    # Create an empty list to collect links for manual evaluation and an empty dataframe to collect results
    manual_links = []
    query_ratings = pd.DataFrame(columns = ['domain', 'query', 'url',
                                            'domain_trust', 'content_relevance',
                                            'title_relevance', 'citation_score',
                                            'final_score'])

    query_num = 0
    # Loop through the queries to generate and evaluate links from SerpAPI
    for i, domain_query in enumerate(list(zip(knowledge_domains, queries))):
        domain, query = domain_query
        print(query)
        # Search SerpAPI for links related to the query
        print(f"Acquiring search results for query {i + 1}/{len(queries)}.")
        results = search_serpapi(query, api_key, num_results = 40)

        # If an error occurs, print the error message and continue
        if type(results) == str:
            print(results)
            continue

        # Generate an empty dataframe to collect ratings for each url
        link_df = pd.DataFrame()

        # For each url, gather the rating and add to link_df
        for j, link in enumerate(results):
            print(f"Evaluating link {j + 1}/{len(results)} for query {i + 1}/{len(queries)}.")
            citation = bool(domain in ['Health', 'Medicine'])
            ratings = validator.rate_url_validity(query, link,
                                                  flags = {'citation': citation,
                                                           'domain': domain})['raw_scores']
            temp_df = pd.DataFrame.from_dict({k: [v] for k, v in ratings.items()})
            temp_df.insert(0, 'domain', domain)
            temp_df.insert(1, 'query', query)
            temp_df.insert(2, 'url', link)
            link_df = pd.concat([link_df, temp_df], ignore_index = True)

        # For each query, randomly select one of the collected links
        manual_links.append(np.random.choice(results))

        # Add ratings for the query to top-level dataframe
        query_ratings = pd.concat([query_ratings, link_df], ignore_index = True)
        
        # Fill null values with 0 and write to csv to preserve inermediate progress
        temp_path = f'./testing/unit_tests-queries_01-{i + 1:02d}.csv'
        query_ratings = query_ratings.fillna(0)
        query_ratings.to_csv(temp_path, index = False)
        print(f'Saved intermediate results to {temp_path}.')

    # Fill null values with 0 and write to csv
    query_ratings = query_ratings.fillna(0)
    query_ratings.to_csv('./testing/unit_tests.csv', index = False)

    # Extract one sample link per query, format, and write to csv
    manual_ratings = query_ratings.loc[query_ratings.url.isin(manual_links)][['query', 'url', 'final_score']]
    manual_ratings.columns = ['user_prompt', 'url_to_check', 'func_rating']
    manual_ratings['custom_rating'] = 0
    manual_ratings.to_csv('./testing/sample.csv', index = False)

    return

if __name__ == '__main__':
    csv_path = './testing/test_queries.csv'
    api_key = serpapi_key
    process_csv(csv_path, api_key)
    