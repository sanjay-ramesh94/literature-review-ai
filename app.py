
from flask import Flask, request, jsonify
from flask_cors import CORS
from semantic_api import search_papers
import logging
import requests
import os
import PyPDF2
import io
import re
import time
import random
import json
import asyncio
import aiohttp
from cachetools import TTLCache
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# Set logging to DEBUG temporarily for detailed diagnostics
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
DEFAULT_MODEL = "gemini-1.5-flash"

search_cache = TTLCache(maxsize=100, ttl=3600)
text_cache = TTLCache(maxsize=1000, ttl=3600)

executor = ThreadPoolExecutor(max_workers=10)

def test_gemini_api(model=DEFAULT_MODEL):
    if not GEMINI_API_KEY:
        app.logger.error("GEMINI_API_KEY is not set in .env")
        return False
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    payload = {
        "contents": [{"parts": [{"text": "Test"}]}],
        "generationConfig": {"maxOutputTokens": 10}
    }
    try:
        response = requests.post(
            GEMINI_API_URL,
            json=payload,
            headers=headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        app.logger.info(f"Gemini API test successful with model: {model}")
        return True
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Gemini API test failed for model {model}: {e}")
        return False

async def fetch_pdf_content(session, pdf_url):
    try:
        async with session.get(pdf_url, timeout=30) as response:
            if response.status != 200 or 'application/pdf' not in response.headers.get('Content-Type', '').lower():
                app.logger.debug(f"Invalid or inaccessible PDF URL: {pdf_url} (Status: {response.status})")
                return None
            content = await response.read()
            app.logger.debug(f"Successfully fetched PDF content from {pdf_url} ({len(content)} bytes)")
            return content
    except Exception as e:
        app.logger.debug(f"Error fetching PDF {pdf_url}: {e}")
        return None

def preprocess_pdf_text(text, query=None):
    if not text:
        app.logger.debug("No text provided for preprocessing")
        return None
    try:
        paragraphs = text.split('\n\n')
        processed_text = []
        in_references = False
        in_appendix = False
        section_headers = [
            'abstract', 'introduction', 'method', 'methodology', 'results',
            'discussion', 'conclusion', 'experiments', 'evaluation'
        ]
        skip_headers = ['references', 'bibliography', 'acknowledgments', 'appendix', 'supplementary']
        query_keywords = [q.lower() for q in query.split()] if query else []

        for para in paragraphs:
            para_lower = para.lower().strip()
            if len(para.strip()) < 20:
                continue
            first_line = para_lower.split('\n')[0]
            if any(first_line.startswith(header) for header in skip_headers):
                if 'references' in first_line or 'bibliography' in first_line:
                    in_references = True
                elif 'appendix' in first_line or 'supplementary' in first_line:
                    in_appendix = True
                continue
            elif any(first_line.startswith(header) for header in section_headers):
                in_references = False
                in_appendix = False
            if in_references or in_appendix:
                continue
            if query_keywords and any(kw in para_lower for kw in query_keywords):
                processed_text.append(para)
            elif not query_keywords:
                processed_text.append(para)

        cleaned_text = ' '.join(processed_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        max_length = 20000
        if len(cleaned_text) > max_length:
            app.logger.debug(f"Truncating text from {len(cleaned_text)} to {max_length} characters")
            cleaned_text = cleaned_text[:max_length]
        if len(cleaned_text) < 50:
            app.logger.debug("Preprocessed text too short")
            return None
        app.logger.debug(f"Preprocessed text length: {len(cleaned_text)} characters")
        return cleaned_text
    except Exception as e:
        app.logger.error(f"Error during text preprocessing: {e}")
        return text[:20000] if text else None

def extract_text_from_pdf(pdf_file, is_url=True, query=None):
    cache_key = pdf_file if is_url else None
    if cache_key and cache_key in text_cache:
        app.logger.debug(f"Returning cached text for {pdf_file}")
        return text_cache[cache_key]

    try:
        if is_url:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            async def fetch():
                async with aiohttp.ClientSession() as session:
                    content = await fetch_pdf_content(session, pdf_file)
                    return content
            pdf_data = loop.run_until_complete(fetch())
            loop.close()
            if not pdf_data:
                app.logger.debug(f"No PDF content retrieved for {pdf_file}")
                return None
            pdf_data = io.BytesIO(pdf_data)
        else:
            pdf_data = pdf_file

        reader = PyPDF2.PdfReader(pdf_data)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n\n"

        text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
        if not text:
            app.logger.debug(f"No text extracted from {pdf_file}")
            return None
        processed_text = preprocess_pdf_text(text, query)

        if processed_text and cache_key:
            text_cache[cache_key] = processed_text
        elif not processed_text:
            app.logger.debug(f"Failed to extract meaningful text from {pdf_file}")
        return processed_text
    except Exception as e:
        app.logger.error(f"PDF extraction failed for {pdf_file}: {e}")
        return None

async def summarize_with_gemini_async(text, title, model=DEFAULT_MODEL):
    if not text or len(text) < 50:
        app.logger.debug(f"Text too short for summarization: {len(text)} chars for {title}")
        return "Summary unavailable - insufficient content"

    if not GEMINI_API_KEY:
        app.logger.error("GEMINI_API_KEY is not set")
        return "Summary unavailable - API key missing"

    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    prompt = f"""You are a research analyst. Analyze this paper and create a structured summary focusing on:
1. Core research problem and objectives
2. Key methodologies/techniques used
3. Main findings/results
4. Significant contributions
5. Potential applications
6. Limitations
Format as concise bullet points. Use academic language but avoid jargon. Paper title: {title}

Paper content:
{text[:20000]}"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 600
        }
    }

    async def attempt_request(session, attempt):
        try:
            app.logger.debug(f"Summarizing attempt {attempt+1} for title: {title}")
            async with session.post(GEMINI_API_URL, json=payload, headers=headers, params=params, timeout=60) as response:
                if response.status == 400:
                    error_detail = await response.text()
                    app.logger.error(f"Attempt {attempt+1} failed: 400 Bad Request: {error_detail}")
                    return None, "Invalid request"
                if response.status == 429:
                    app.logger.warning(f"Attempt {attempt+1} failed: 429 Too Many Requests")
                    return None, "Rate limit exceeded"
                if response.status == 401:
                    app.logger.error(f"Attempt {attempt+1} failed: Invalid API key")
                    return "Summary unavailable - invalid API key", None
                if response.status == 503:
                    app.logger.error(f"Attempt {attempt+1} failed: 503 Service Unavailable")
                    return None, "Service unavailable"
                response.raise_for_status()
                data = await response.json()
                summary = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if not summary:
                    app.logger.warning(f"No summary content for {title}")
                    return "Summary unavailable - empty API response", None
                app.logger.debug(f"Summary generated for {title}")
                return summary, None
        except aiohttp.ClientError as e:
            app.logger.error(f"Attempt {attempt+1} failed for {title}: {e}")
            return None, str(e)

    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            summary, error = await attempt_request(session, attempt)
            if summary:
                return summary
            if error in ["Invalid request", "invalid API key"]:
                return summary or f"Summary unavailable - API error ({error})"
            if error == "Rate limit exceeded" and attempt == 2:
                return "Summary unavailable - API rate limit exceeded"
            if error == "Service unavailable" and attempt == 2:
                return "Summary unavailable - API service unavailable"
            await asyncio.sleep(2 + random.uniform(0, 1))
        return "Summary unavailable - repeated API failures"

@app.route('/')
def home():
    app.logger.info("Root endpoint '/' accessed.")
    return "Literature Review AI Backend is running!"

@app.route('/test_search')
def test_search():
    try:
        papers = search_papers("test", limit=1)
        return jsonify({"status": "success", "papers_found": len(papers or [])})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/search', methods=['POST'])
async def handle_search():
    if not request.is_json:
        app.logger.warning("Request without JSON payload.")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query')
    limit = data.get('limit', 100)
    summarize = data.get('summarize', False)

    if not query:
        app.logger.warning("Missing 'query' in request.")
        return jsonify({"error": "Missing 'query' in request data"}), 400

    if not isinstance(limit, int) or limit <= 0:
        app.logger.warning(f"Invalid 'limit': {limit}")
        return jsonify({"error": "'limit' must be a positive integer"}), 400

    require_pdf_filter = request.args.get('require_pdf', 'false').lower() == 'true'
    app.logger.info(f"Search: query='{query}', limit={limit}, require_pdf={require_pdf_filter}, summarize={summarize}")

    cache_key = f"{query}:{limit}:{require_pdf_filter}"
    if cache_key in search_cache:
        app.logger.info(f"Returning cached results for {cache_key}")
        return jsonify(search_cache[cache_key])

    try:
        papers = search_papers(query, limit=limit * 2 if require_pdf_filter else limit)
    except Exception as e:
        app.logger.error(f"Error calling search_papers: {e}", exc_info=True)
        return jsonify({"error": "Could not connect to search service", "details": str(e)}), 503

    if papers is None or not papers:
        app.logger.error(f"Failed to retrieve papers for query '{query}'")
        return jsonify({"error": "Search service unavailable or no papers found"}), 503

    processed_results = []
    papers_added = 0

    async def process_paper(paper):
        pdf_url = paper.get('pdfUrl')
        title = paper.get('title', 'Untitled')
        paper_id = paper.get('paperId', 'N/A')
        abstract = paper.get('abstract', '')

        if require_pdf_filter and not pdf_url:
            app.logger.debug(f"Skipping {paper_id} - no PDF available")
            return None

        # Early check for usable content
        if not pdf_url and not abstract:
            app.logger.debug(f"Skipping {paper_id} - no PDF or abstract available")
            return None

        try:
            authors_list = paper.get('authors', [])
            author_names = [
                author.get('name') for author in authors_list if author and author.get('name')
            ] if isinstance(authors_list, list) else []

            essential_data = {
                "title": title,
                "abstract": abstract,
                "year": paper.get('year'),
                "authors": author_names,
                "pdfUrl": pdf_url,
                "summary": None,
                "paperId": paper_id
            }

            if summarize:
                try:
                    full_text = None
                    if pdf_url:
                        full_text = extract_text_from_pdf(pdf_url, query=query)
                    summary_source = full_text if full_text else abstract

                    if not summary_source:
                        app.logger.debug(f"No content (PDF or abstract) for {paper_id}")
                        essential_data['summary'] = "Summary unavailable - no content available"
                    else:
                        summary = await summarize_with_gemini_async(summary_source, title)
                        if summary and not summary.startswith("Summary unavailable") and not full_text:
                            summary += "\n(Summary based on abstract only - full text unavailable)"
                        essential_data['summary'] = summary
                except Exception as e:
                    app.logger.error(f"Summarization failed for {paper_id}: {e}", exc_info=True)
                    essential_data['summary'] = f"Summary unavailable - processing error: {str(e)}"

            return essential_data
        except Exception as e:
            app.logger.error(f"Error processing paper {paper_id}: {e}", exc_info=True)
            return None

    tasks = [process_paper(paper) for paper in papers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if result and not isinstance(result, Exception) and papers_added < limit:
            processed_results.append(result)
            papers_added += 1
        elif isinstance(result, Exception):
            app.logger.error(f"Task failed with exception: {result}")

    search_cache[cache_key] = processed_results
    app.logger.info(f"Returning {len(processed_results)} papers for query '{query}'")
    return jsonify(processed_results)

@app.route('/upload', methods=['POST'])
async def handle_upload():
    if 'files' not in request.files:
        app.logger.warning("No files in upload request")
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist('files')
    results = []

    async def process_file(file):
        if file.mimetype != 'application/pdf':
            app.logger.warning(f"Skipping non-PDF file: {file.filename}")
            return None

        try:
            text = extract_text_from_pdf(file.stream, is_url=False)
            summary = None
            if text:
                summary = await summarize_with_gemini_async(text, file.filename)
            else:
                app.logger.debug(f"No text extracted from {file.filename}")
                summary = "Summary unavailable - failed to extract text"

            return {
                "title": file.filename,
                "abstract": None,
                "year": None,
                "authors": [],
                "pdfUrl": None,
                "summary": summary,
                "paperId": None
            }
        except Exception as e:
            app.logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
            return {
                "title": file.filename,
                "summary": f"Summary unavailable - processing error: {str(e)}"
            }

    tasks = [process_file(file) for file in files]
    file_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in file_results:
        if result and not isinstance(result, Exception):
            results.append(result)
        elif isinstance(result, Exception):
            app.logger.error(f"Upload task failed with exception: {result}")

    app.logger.info(f"Processed {len(results)} uploaded files")
    return jsonify(results)

if __name__ == '__main__':
    app.logger.info("Starting Flask development server...")
    if test_gemini_api(DEFAULT_MODEL):
        app.logger.info("Gemini API is accessible")
    app.run(debug=True, host='0.0.0.0', port=5000)