import argparse
import os
import re
import sys
import requests
import logging
from semantic_api import search_papers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_OUTPUT_DIR = "downloaded_papers"
DOWNLOAD_TIMEOUT = 60

def sanitize_filename(filename):
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', filename)
    sanitized = sanitized.replace(' ', '_')
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_.')
    return sanitized[:150]

def download_pdf(pdf_url: str, save_path: str) -> bool:
    try:
        if os.path.exists(save_path):
            logging.info(f"Skipping download, file exists: '{os.path.basename(save_path)}'")
            return True

        logging.info(f"Downloading from: {pdf_url}")
        response = requests.get(pdf_url, stream=True, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' not in content_type:
            logging.warning(f"URL did not return PDF (got '{content_type}'). Skipping")
            response.close()
            return False

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logging.info(f"Downloaded: '{os.path.basename(save_path)}'")
        return True

    except requests.exceptions.Timeout:
        logging.error(f"Timeout downloading {pdf_url}")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Download failed for {pdf_url}: {e}")
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError as e:
                logging.error(f"Could not remove incomplete file: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error downloading {pdf_url}: {e}")
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError as e:
                logging.error(f"Could not remove incomplete file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Search Semantic Scholar and download PDFs")
    parser.add_argument("query", type=str, help="Search query for papers")
    parser.add_argument("-l", "--limit", type=int, default=10, help="Max papers to retrieve (default: 10)")
    parser.add_argument("-o", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: '{DEFAULT_OUTPUT_DIR}')")
    parser.add_argument("-d", "--max-downloads", type=int, default=None, help="Max PDFs to download (default: all available)")

    args = parser.parse_args()

    try:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory: '{output_dir}'")
    except OSError as e:
        logging.error(f"Failed to create output directory: {e}")
        sys.exit(1)

    logging.info(f"Searching for: '{args.query}' (limit: {args.limit})")
    papers = search_papers(query=args.query, limit=args.limit)

    if papers is None:
        logging.error("Failed to retrieve papers from API")
        sys.exit(1)
    elif not papers:
        logging.info("No papers found")
        sys.exit(0)

    successful_downloads = 0
    papers_with_pdf = 0

    for i, paper in enumerate(papers):
        if args.max_downloads is not None and successful_downloads >= args.max_downloads:
            logging.info(f"Reached download limit ({args.max_downloads})")
            break

        title = paper.get('title', f'untitled_{i}')
        year = paper.get('year', 'N/A')
        pdf_url = paper.get('pdfUrl')

        log_prefix = f"[{i+1}/{len(papers)}] '{title[:50]}...' ({year})"

        if pdf_url:
            papers_with_pdf += 1
            safe_title = sanitize_filename(title)
            filename = f"{year}_{safe_title}_{paper.get('paperId', '')[:8]}.pdf"
            save_path = os.path.join(output_dir, filename)

            if download_pdf(pdf_url, save_path):
                successful_downloads += 1
            else:
                logging.warning(f"{log_prefix} Download failed")
        else:
            logging.info(f"{log_prefix} No PDF available")

    print("\n" + "="*50)
    print("Download Summary")
    print("="*50)
    print(f"Query:           '{args.query}'")
    print(f"Papers found:    {len(papers)}")
    print(f"Papers with PDF: {papers_with_pdf}")
    print(f"Downloads:       {successful_downloads}")
    print(f"Saved in:       '{output_dir}'")
    print("="*50)

if __name__ == "__main__":
    main()