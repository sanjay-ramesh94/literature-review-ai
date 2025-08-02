# download_papers.py
import argparse
import os
import re
import sys
import requests
import logging
# Import the function from your *updated* semantic_searcher.py
from semantic_api import search_papers

# Setup logging consistent with semantic_searcher
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DEFAULT_OUTPUT_DIR = "downloaded_papers"
DOWNLOAD_TIMEOUT = 60 # Timeout for downloading a single file in seconds
# ---------------------

def sanitize_filename(filename):
    """Removes or replaces characters illegal in most file systems."""
    # Remove illegal characters: < > : " / \ | ? *
    # Also remove control characters (like newline, tab etc.)
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Reduce multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores/periods
    sanitized = sanitized.strip('_.')
    # Limit length (optional, e.g., to 150 chars to avoid issues)
    return sanitized[:150]

def download_pdf(pdf_url: str, save_path: str) -> bool:
    """
    Attempts to download a PDF from a given URL and save it.

    Args:
        pdf_url: The direct URL to the PDF file (obtained from paper['pdfUrl']).
        save_path: The full path where the PDF should be saved.

    Returns:
        True if download was successful, False otherwise.
    """
    try:
        # Check if file already exists before attempting download
        if os.path.exists(save_path):
            logging.info(f"Skipping download, file already exists: '{os.path.basename(save_path)}'")
            return True # Consider it a success if the file is already there

        logging.info(f"Attempting download from: {pdf_url}")
        response = requests.get(pdf_url, stream=True, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True)
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

        # Check Content-Type header more robustly
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' not in content_type:
            logging.warning(f"URL did not return PDF content-type (got '{content_type}'). Skipping download for '{os.path.basename(save_path)}'")
            # Consider logging the first few bytes if debugging content issues:
            # logging.debug(f"First 100 bytes of content: {response.content[:100]}")
            response.close() # Close the connection
            return False

        # Download the file chunk by chunk
        logging.info(f"Downloading to: '{os.path.basename(save_path)}'")
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                # filter out keep-alive new chunks
                if chunk:
                    f.write(chunk)
        logging.info(f"Successfully downloaded: '{os.path.basename(save_path)}'")
        return True

    except requests.exceptions.Timeout:
        logging.error(f"Timeout error while downloading {pdf_url}")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Download failed for {pdf_url}: {e}")
        # Clean up potentially incomplete file if download failed partway
        if os.path.exists(save_path):
             try:
                 os.remove(save_path)
                 logging.info(f"Removed incomplete file: '{os.path.basename(save_path)}'")
             except OSError as rm_err:
                 logging.error(f"Could not remove incomplete file {save_path}: {rm_err}")
        return False
    except Exception as e: # Catch any other unexpected errors during download/saving
        logging.error(f"An unexpected error occurred for {pdf_url}: {e}", exc_info=True)
        if os.path.exists(save_path):
             try:
                 os.remove(save_path)
                 logging.info(f"Removed incomplete file: '{os.path.basename(save_path)}'")
             except OSError as rm_err:
                 logging.error(f"Could not remove incomplete file {save_path}: {rm_err}")
        return False


def main():
    """
    Main function to parse arguments, search for papers, and attempt downloads.
    """
    parser = argparse.ArgumentParser(
        description="Search Semantic Scholar and download available open access PDFs using semantic_searcher.py"
    )
    parser.add_argument("query", type=str, help="Search query for papers.")
    parser.add_argument(
        "-l", "--limit", type=int, default=10,
        help="Max number of paper *details* to retrieve via API (default: 10)."
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save downloaded PDFs (default: '{DEFAULT_OUTPUT_DIR}')."
    )
    parser.add_argument(
        "-d", "--max-downloads", type=int, default=None,
        help="Maximum number of PDFs to actually download (optional, downloads all available by default)."
    )

    args = parser.parse_args()

    # --- 1. Create Output Directory ---
    try:
        # Use os.path.abspath for clearer log messages later
        output_directory = os.path.abspath(args.output_dir)
        os.makedirs(output_directory, exist_ok=True)
        logging.info(f"Using output directory: '{output_directory}'")
    except OSError as e:
        logging.error(f"Fatal: Failed to create output directory '{args.output_dir}': {e}")
        sys.exit(1)

    # --- 2. Search for Papers using the updated semantic_searcher ---
    logging.info(f"Searching Semantic Scholar for '{args.query}' (API limit {args.limit})...")
    # search_papers now returns papers with a 'pdfUrl' key if available
    papers = search_papers(query=args.query, limit=args.limit)

    if papers is None:
        logging.error("Fatal: Failed to retrieve paper details from Semantic Scholar API.")
        sys.exit(1)
    elif not papers:
        logging.info("No papers found matching the query.")
        sys.exit(0)

    logging.info(f"API returned details for {len(papers)} paper(s).")

    # --- 3. Attempt Downloads ---
    successful_downloads = 0
    papers_with_link = 0
    papers_processed = 0

    for i, paper in enumerate(papers):
        papers_processed += 1
        # Stop if max downloads reached (based on successful downloads)
        if args.max_downloads is not None and successful_downloads >= args.max_downloads:
            logging.info(f"Reached download limit ({args.max_downloads}). Stopping further download attempts.")
            break

        # --- Extract relevant info ---
        paper_id = paper.get('paperId', f'unknown_id_{i}')
        title = paper.get('title', f'untitled_paper_{i}')
        year = paper.get('year', 'N/A')
        # *** Get the pre-processed PDF URL from semantic_searcher ***
        pdf_url = paper.get('pdfUrl') # This relies on your updated semantic_searcher.py

        log_prefix = f"[Paper {i+1}/{len(papers)}] '{title[:60]}...' ({year})"
        logging.info(f"\n{log_prefix}: Processing...")

        if pdf_url:
            papers_with_link += 1
            logging.info(f"{log_prefix}: Found potential PDF link: {pdf_url}")

            # Create a safe filename
            safe_title = sanitize_filename(title)
            # Add year and part of ID for uniqueness and context
            filename = f"{year}_{safe_title}_{paper_id[:8]}.pdf"
            save_path = os.path.join(output_directory, filename)

            # Call the download function
            if download_pdf(pdf_url, save_path):
                successful_downloads += 1
            else:
                # download_pdf function already logs the failure reason
                logging.warning(f"{log_prefix}: Download attempt failed.")
        else:
            logging.info(f"{log_prefix}: No direct PDF link ('pdfUrl') found in API data.")

    # --- 4. Final Summary ---
    print("\n" + "="*50)
    print("Download Task Summary")
    print("="*50)
    print(f"Search Query:          '{args.query}'")
    print(f"API Request Limit:     {args.limit}")
    print(f"Paper Details Found:   {len(papers)}")
    print(f"Papers Processed:      {papers_processed}")
    print(f"Papers with PDF Link:  {papers_with_link}")
    print(f"Download Limit Set:    {'None (All available)' if args.max_downloads is None else args.max_downloads}")
    print(f"Successful Downloads:  {successful_downloads}")
    print(f"Files saved in:        '{output_directory}'")
    print("="*50)

if __name__ == "__main__":
    main()