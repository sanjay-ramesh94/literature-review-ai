# semantic_api.py
import requests
import os
import time
import random
import logging
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
# SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1
BACKOFF_FACTOR = 2

def search_papers(query: str, limit: int = 10) -> list | None:
    """
    Searches for papers using the Semantic Scholar API, then manually sorts
    results by citation count descending. Includes retries and PDF link retrieval.

    Args:
        query: The search string.
        limit: The maximum number of papers to return.

    Returns:
        A list of paper details sorted by citation count desc, or None on failure.
    """
    endpoint = f"{SEMANTIC_SCHOLAR_API_URL}/paper/search"

    # --- Parameters: REMOVED 'sort', kept 'citationCount' in 'fields' ---
    params = {
        'query': query,
        'limit': limit,
        # Still need citationCount to sort manually
        'fields': 'paperId,url,title,abstract,authors,year,isOpenAccess,openAccessPdf,citationCount',
        # 'sort': 'citationCount:desc' # <-- REMOVED API sorting parameter
    }
    headers = {  }
    current_backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(MAX_RETRIES + 1):
        try:
            logging.info(f"Attempt {attempt+1}/{MAX_RETRIES+1}: Querying S2 API (Manual Sort Planned): query='{query}', limit={limit}")
            prepared_request = requests.Request('GET', endpoint, params=params, headers=headers).prepare()
            logging.debug(f"Requesting URL: {prepared_request.url}")

            response = requests.get(endpoint, params=params, headers=headers, timeout=30)

            # --- Handle Retries (same as before) ---
            if response.status_code == 429: # ... (retry logic) ...
                 logging.warning(f"Attempt {attempt+1} failed: 429 Too Many Requests.")
                 if attempt == MAX_RETRIES: break
                 wait_time = current_backoff + random.uniform(0, 0.5); logging.info(f"Waiting {wait_time:.2f}s..."); time.sleep(wait_time); current_backoff *= BACKOFF_FACTOR; continue
            if response.status_code >= 500: # ... (retry logic) ...
                 logging.warning(f"Attempt {attempt+1} failed: Server Error {response.status_code}.")
                 if attempt == MAX_RETRIES: break
                 wait_time = current_backoff + random.uniform(0, 0.5); logging.info(f"Waiting {wait_time:.2f}s..."); time.sleep(wait_time); current_backoff *= BACKOFF_FACTOR; continue

            response.raise_for_status()

            # --- Process successful response ---
            results = response.json()
            papers_found_raw = results.get('data', [])
            total_results = results.get('total', len(papers_found_raw))
            logging.info(f"Attempt {attempt+1} successful. Received {len(papers_found_raw)} raw results (Total matching: {total_results}).")

            # --- Process each paper (same as before) ---
            processed_papers = []
            for paper in papers_found_raw:
                if not isinstance(paper, dict): continue
                processed_paper = {
                    'paperId': paper.get('paperId'), 'url': paper.get('url'), 'title': paper.get('title'),
                    'abstract': paper.get('abstract'), 'authors': paper.get('authors'), 'year': paper.get('year'),
                    'isOpenAccess': paper.get('isOpenAccess'), 'citationCount': paper.get('citationCount'),
                    'pdfUrl': None
                }
                pdf_info = paper.get('openAccessPdf')
                if pdf_info and isinstance(pdf_info, dict) and pdf_info.get('url'):
                    processed_paper['pdfUrl'] = pdf_info['url']
                processed_papers.append(processed_paper)

            # *** ADDED MANUAL SORTING STEP ***
            try:
                # Sort the list of dictionaries in-place
                processed_papers.sort(
                    # Key function: extract citationCount, treat None as -1 (goes last in desc sort)
                    key=lambda p: p.get('citationCount') if p.get('citationCount') is not None else -1,
                    reverse=True # Sort descending (highest citation count first)
                )
                logging.info(f"Manually sorted {len(processed_papers)} papers by citation count (desc).")
            except TypeError as te:
                # Handle potential type errors if citationCount isn't numeric (should be, but defensive)
                 logging.error(f"TypeError during manual sorting: {te}. Returning results unsorted.", exc_info=True)
            except Exception as e:
                 logging.error(f"Unexpected error during manual sorting: {e}. Returning results unsorted.", exc_info=True)
            # *** END MANUAL SORTING STEP ***

            return processed_papers # Return the (now sorted) list

        except requests.exceptions.Timeout: # ... (timeout handling) ...
             logging.warning(f"Attempt {attempt+1} timed out.")
             if attempt == MAX_RETRIES: break
             wait_time = current_backoff + random.uniform(0, 0.5); logging.info(f"Waiting {wait_time:.2f}s..."); time.sleep(wait_time); current_backoff *= BACKOFF_FACTOR; continue
        except requests.exceptions.RequestException as e: # ... (other request error handling) ...
             logging.error(f"Attempt {attempt+1} failed with RequestException: {e}", exc_info=True)
             return None

    # Failure after all retries
    logging.error(f"Failed to retrieve papers for query '{query}' after {MAX_RETRIES+1} attempts.")
    return None

# --- Example Usage (Corrected, same as before) ---
if __name__ == '__main__':
    test_query = "large language models for code generation"
    print(f"\nTesting semantic_api.py with query: '{test_query}' (Manual Sort)")
    print("-" * 30)
    papers_result = search_papers(test_query, limit=5)
    print("-" * 30)
    if papers_result is None: print(f"[RESULT] Failed to retrieve papers for '{test_query}' after retries.")
    elif not papers_result: print(f"[RESULT] Successfully queried, but found 0 papers for '{test_query}'.")
    else:
        print(f"[RESULT] Found {len(papers_result)} papers for '{test_query}' (MANUALLY sorted by citation count desc):")
        for i, paper in enumerate(papers_result):
            # ...(same printing logic as before, displaying citation count)...
            title = paper.get('title', 'N/A'); year = paper.get('year', 'N/A'); citations = paper.get('citationCount', 'N/A')
            is_oa = paper.get('isOpenAccess', False); pdf_url = paper.get('pdfUrl', 'Not Available')
            authors_list = paper.get('authors', []); author_names = [a.get('name', '?') for a in authors_list if isinstance(a, dict)] if isinstance(authors_list, list) else []
            print(f"\n  {i+1}. {title}"); print(f"     Year: {year} | Citations: {citations}"); print(f"     Authors: {', '.join(author_names) if author_names else 'N/A'}"); print(f"     Open Access: {is_oa}"); print(f"     PDF Link: {pdf_url}")
    print("\nScript finished.")