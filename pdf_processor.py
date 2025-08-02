from flask import Flask, request, jsonify
from flask_cors import CORS
from semantic_api import search_papers
import logging
import os
import asyncio
import aiohttp
from cachetools import TTLCache
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from io import BytesIO
import requests
import PyPDF2
import re
from pathlib import Path

load_dotenv()

# Set logging to DEBUG temporarily for detailed diagnostics
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
CORS(app)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)
LLM_MODEL_NAME = 'gemini-1.5-flash'
llm_model = genai.GenerativeModel(LLM_MODEL_NAME)
app.logger.info(f"Gemini client configured successfully for model '{LLM_MODEL_NAME}'.")

search_cache = TTLCache(maxsize=100, ttl=3600)
text_cache = TTLCache(maxsize=1000, ttl=3600)

executor = ThreadPoolExecutor(max_workers=10)

def extract_text_from_pdf(pdf_file, is_url=True):
    try:
        if is_url:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            async def fetch():
                async with aiohttp.ClientSession() as session:
                    async with session.get(pdf_file, timeout=30) as response:
                        if response.status != 200 or 'application/pdf' not in response.headers.get('Content-Type', '').lower():
                            app.logger.debug(f"Invalid or inaccessible PDF URL: {pdf_file} (Status: {response.status})")
                            return None
                        return await response.read()
            pdf_data = loop.run_until_complete(fetch())
            loop.close()
            if not pdf_data:
                app.logger.debug(f"No PDF content retrieved for {pdf_file}")
                return None
            pdf_data = BytesIO(pdf_data)
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
        return text
    except Exception as e:
        app.logger.error(f"PDF extraction failed for {pdf_file}: {e}")
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    """ Splits text into overlapping chunks based on word count. """
    if not text: return []
    words = text.split()
    if not words: return []

    chunks = []
    start_index = 0
    while start_index < len(words):
        end_index = min(start_index + chunk_size, len(words))
        chunk_words = words[start_index:end_index]
        chunks.append(" ".join(chunk_words))

        next_start = start_index + chunk_size - overlap
        if next_start <= start_index:
            next_start = start_index + 1
        if next_start >= len(words):
             break
        start_index = next_start

    filtered_chunks = [chunk for chunk in chunks if chunk and not chunk.isspace()]
    app.logger.info(f"Split text into {len(filtered_chunks)} chunks.")
    return filtered_chunks

async def summarize_with_gemini(text, title):
    """ Generates a structured summary using Gemini based on provided text. """
    if not text or len(text) < 50:
        app.logger.debug(f"Text too short for summarization: {len(text)} chars for {title}")
        return "Summary unavailable - insufficient content"

    chunks = chunk_text(text)
    if not chunks:
        return "Summary unavailable - no meaningful content found"

    context = "\n\n---\n\n".join(chunks[:10])  # Limit to first 10 chunks
    prompt = f"""
You are an AI assistant tasked with extracting key information from text chunks of a research paper and presenting it in a structured format.

**Input Text:**
The following text consists of extracted sections from a single research paper:
--- START OF EXTRACTED TEXT ---
{context}
--- END OF EXTRACTED TEXT ---

**Instructions:**

Carefully analyze the "Input Text" provided above. Based *strictly and exclusively* on the information contained within that text, generate a structured summary by populating the sections below.

**Output Format Requirements:**
Use the following exact headings. Under each heading, provide a concise summary of the relevant information found *only* in the "Input Text".

**Objective:**
[Analyze the text and summarize the main research question, problem statement, or primary goal discussed. If this information is not clearly present in the text, state: Information not found in provided text.]

**Methodology:**
[Analyze the text and describe the core methods, experimental procedures, datasets, or analytical approaches mentioned. If this information is not clearly present in the text, state: Information not found in provided text.]

**Key Findings:**
[Analyze the text and list or summarize the most significant results, discoveries, or key outcomes reported. If this information is not clearly present in the text, state: Information not found in provided text.]

**Conclusion/Implications:**
[Analyze the text and summarize the main conclusions drawn by the authors, along with any stated implications, contributions, or limitations mentioned. If this information is not clearly present in the text, state: Information not found in provided text.]

**Important Constraints:**
* Present the output using the specified headings ONLY.
* Extract information *solely* from the provided "Input Text". Do not infer, add external knowledge, or interpret beyond what is written.
* Keep the summary under each heading concise and objective.
* If relevant information for a heading is absent in the text, explicitly use the phrase "Information not found in provided text". Do not leave sections blank or attempt to fill them without textual evidence.
"""

    try:
        response = await asyncio.to_thread(llm_model.generate_content, prompt)
        review_text = response.text
        app.logger.info("Review generated successfully via Gemini.")
        return review_text
    except Exception as e:
        app.logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        return "Error: Could not generate the review due to an issue with the AI service."

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
                        full_text = extract_text_from_pdf(pdf_url)
                    summary_source = full_text if full_text else abstract

                    if not summary_source:
                        app.logger.debug(f"No content (PDF or abstract) for {paper_id}")
                        essential_data['summary'] = "Summary unavailable - no content available"
                    else:
                        summary = await summarize_with_gemini(summary_source, title)
                        if summary and not summary.startswith("Summary unavailable") and not full_text:
                            summary += "\n"
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
                summary = await summarize_with_gemini(text, file.filename)
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
    app.run(debug=True, host='0.0.0.0', port=5000)