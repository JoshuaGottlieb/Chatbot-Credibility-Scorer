# Algorithm Analysis of rate_url_validity() function by ChatGPT

## Step 1: Scraping the URL Content

The scraping process only extracts text from portions of the page wrapped in paragraph (\<p>\</p>) tags. This assumes that the webpage is set up so that all of the relevant content is in this format, as opposed to a more general HTML element such as div tags. While grabbing more text could lead to some issues, there are many examples where this scraping process fails to extract useful information from the webpage, as evidenced by [the unit tests](unit_tests.ipynb). The process also terminates completely if the page is unable to be scraped, when there are two aspects (domain trustworthiness and citation score) that can be evaluated solely on the webpage URL without needing to access the webpage content.

To fix this problem, the scraper should be adjusted to be more generous in grabbing text from the webpage.

## Step 2: Domain Trustworthiness

This section currently is set up with a placeholder value, and thus needs to be changed to some method that is useful and real. Preferably, this should not rely on using a paid API such as Moz for ease of testing and ability to be used for free.

## Step 3: Content Relevance

This section uses a sentence transformer from Hugging Face. This method appears to be sound but may need tweaking. As evidenced in [the unit tests](unit_tests.ipynb), this function appears to heavily penalize webpages with lots of text. The scores returned are fairly low (<75) even for webpages with content that highly matches the supplied user query, which gives the illusion that sources are not as relevant as they actually are. It is also not ideal that there can be websites that are highly relevant with low/no text (and thus are given low content relevance scores by the current function), such as when searching for stock prices, since the semantic similarity scorer fails to perform well when no text is given.

## Step 4: Fact Checking

The template given by ChatGPT references a malformed API endpoint for the Google Fact Check API. Therefore, it fails every time. In addition, the scoring system used is overly simplistic with only three possible outputs (80, 50, 40) which gives no real distinction between the quality of different webpages. Even if the query to the Google Fact Check API was properly formed, the query only uses the first 200 characters from the scraped webpage for no discernible reason. This setup is laughably bad and seems essentially useless for the purposes of credibility scoring.

Upon further research, the Google Fact Check API has no real innate credibility. Any person can add "facts", and those facts are not guaranteed to be verified. So, this metric for credibility scoring seems unnecessary and should likely be dropped.

## Step 5: Bias Detection (Allegedly)

The "bias detection" section of the algorithm uses an NLP Sentiment Analysis model which outputs "Positive", "Neutral", or "Negative" when given a set of text. This sentiment analysis has absolutely nothing to do with whether a source is biased or not. Bias is likely domain-specific, as there are different types of bias (ex: political bias of news sources, or source bias of research studies) which makes it difficult to properly measure. Even if there is a general method for assessing bias, it would definitely not be using a generic sentiment analysis module.

In addition, the sentiment analysis module uses only the first 512 characters of the webpage content (possibly because that is the limit of the size of a Tweet). Much like the Fact Checking section, the potential scores are overly simplistic (100, 50, 30), and the Sentiment Analysis module appears to always output "Negative" based on unit testing. This module seems difficult or impossible to implement correctly, and the current implementation is most certainly nonsense. Thus, this module should likely be dropped.

## Step 6: Citation Check

This module appears to work correctly, but it is irrelevant for nearly all queries and webpages. Mathematically speaking, the number of webpages that are referenced in academic papers on Google Scholar is such a small percentage as to be effectively zero. Therefore, it should be expected that this part of the scoring algorithm will return a score of 0 in almost all cases. This is not a helpful scoring metric.

Additionally, how many times a webpage appears in research papers is relevant only for queries that specifically require academic resources to be returned. In all other cases, this metric is irrelevant, which means that the weight assigned to this section should be close to zero. For these reasons, it would seem best to drop this section of the scoring algorithm.

## Conclusions

The base function given by ChatGPT is mostly useless. Most of the sections return bogus answers, are difficult or impossible to correct, and/or are not particularly useful for rating the credibility of a webpage. The function should be restricted to two criteria: Domain Trust and Content Relevance. The Domain Trust section needs to be written from the ground-up, and the Content Relevance section should be modified to scale its scoring better and to better accommodate webpages that are high in relevance but lack text. The scraping function is passable but should be updated in order to better extract text and other useful information from the webpage.
