import requests
import os
import time
import random
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1

def search_papers(query: str, limit: int = 10) -> list | None:
    """
    Searches Semantic Scholar API with robust retries and simplified PDF handling.
    """
    endpoint = f"{SEMANTIC_SCHOLAR_API_URL}/paper/search"
    params = {
        'query': query,
        'limit': min(limit, 100),  # API max limit
        'fields': 'paperId,url,title,abstract,authors,year,isOpenAccess,openAccessPdf,citationCount'
    }
    headers = {
        'x-api-key': SEMANTIC_SCHOLAR_API_KEY if SEMANTIC_SCHOLAR_API_KEY else None,
        'User-Agent': 'AIResearchAccelerator/1.0'
    }

    current_backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"Attempt {attempt+1}/{MAX_RETRIES}: Querying S2 API: query='{query}'")
            response = requests.get(endpoint, params=params, headers=headers, timeout=15)

            if response.status_code == 429:
                logging.warning(f"Attempt {attempt+1} failed: 429 Too Many Requests")
                wait_time = current_backoff + random.uniform(0, 0.5)
                logging.info(f"Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                current_backoff *= 2
                continue
            if response.status_code >= 400:
                logging.error(f"Attempt {attempt+1} failed: {response.status_code} {response.text}")
                if response.status_code == 401:
                    return None  # Fatal: Invalid API key
                wait_time = current_backoff + random.uniform(0, 0.5)
                time.sleep(wait_time)
                current_backoff *= 2
                continue

            response.raise_for_status()
            results = response.json()
            papers_found = results.get('data', [])
            logging.info(f"Attempt {attempt+1} successful. Received {len(papers_found)} results")

            processed_papers = []
            for paper in papers_found:
                pdf_url = paper.get('openAccessPdf', {}).get('url') if paper.get('isOpenAccess') else None
                paper['pdfUrl'] = pdf_url
                processed_papers.append(paper)

            return processed_papers

        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                logging.error(f"Exhausted retries for query '{query}'")
                return None
            wait_time = current_backoff + random.uniform(0, 0.5)
            logging.info(f"Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            current_backoff *= 2

    return None