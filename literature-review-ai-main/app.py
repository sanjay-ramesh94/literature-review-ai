# app.py
import os
import logging
from datetime import datetime
from pathlib import Path
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import functions from our local modules
import semantic_api # For paper search
import pdf_processor # For PDF processing and review generation

# --- Flask App Initialization ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Basic Routes ---
@app.route('/')
def home():
    app.logger.info("Root endpoint '/' accessed.")
    return "Literature Review AI Backend is running!"

# --- Paper Search Route ---
@app.route('/search', methods=['POST'])
def handle_search():
    """ Handles paper search via Semantic Scholar API. """
    request_timestamp = datetime.now().isoformat()
    # --- 1. Get JSON Payload ---
    if not request.is_json:
        app.logger.warning(f"[{request_timestamp}] /search - Non-JSON request.")
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    query = data.get('query')
    limit = data.get('limit', 10)
    if not query:
        app.logger.warning(f"[{request_timestamp}] /search - Missing 'query'.")
        return jsonify({"error": "Missing 'query' in request data"}), 400
    if not isinstance(limit, int) or limit <= 0:
         app.logger.warning(f"[{request_timestamp}] /search - Invalid 'limit': {limit}")
         return jsonify({"error": "'limit' must be a positive integer"}), 400

    # --- 2. Get Optional Query Parameter ---
    require_pdf_filter = request.args.get('require_pdf', 'false').lower() == 'true'
    app.logger.info(f"[{request_timestamp}] /search - Request: query='{query}', limit={limit}, require_pdf={require_pdf_filter}")

    # --- 3. Call Semantic API ---
    try:
        # Fetch sorted results including citation count
        papers = semantic_api.search_papers(query, limit=limit * 2 if require_pdf_filter else limit)
    except Exception as e:
        app.logger.error(f"[{request_timestamp}] /search - Error calling semantic_api.search_papers: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during search API call"}), 500

    # --- 4. Handle API Call Failure ---
    if papers is None:
        app.logger.error(f"[{request_timestamp}] /search - Failed search for '{query}' after retries.")
        return jsonify({"error": "Search service failed after retries."}), 503

    # --- 5. Filter and Process Results ---
    processed_results = []
    papers_added = 0
    for paper in papers:
        # semantic_api.search_papers now returns dicts with 'pdfUrl' and 'citationCount'
        pdf_url = paper.get('pdfUrl') # pdfUrl is processed inside semantic_api

        # Apply PDF filter if requested
        if require_pdf_filter and not pdf_url:
            continue # Skip this paper

        # Extract only essential fields, **NOW INCLUDING citationCount**
        try:
            authors_list = paper.get('authors', [])
            author_names = [
                author.get('name') for author in authors_list if author and author.get('name')
            ] if isinstance(authors_list, list) else []

            # *** MODIFIED essential_data ***
            essential_data = {
                "title": paper.get('title'),
                "abstract": paper.get('abstract'),
                "year": paper.get('year'),
                "authors": author_names,
                "pdfUrl": pdf_url, # Already processed in semantic_api
                "citationCount": paper.get('citationCount') # *** ADDED THIS LINE ***
            }
            # *** End of modification ***

            processed_results.append(essential_data)
            papers_added += 1

            # Stop adding papers if we reach the original requested limit *after* filtering
            if papers_added >= limit:
                break

        except Exception as e:
            # Log if processing a specific paper fails, but continue with others
            paper_id = paper.get('paperId', 'N/A') # Use paperId if available for logging
            app.logger.error(f"[{request_timestamp}] /search - Error processing essential fields for paper {paper_id}: {e}", exc_info=True)
            continue # Skip faulty paper data


    # --- 6. Return Processed Results ---
    app.logger.info(f"[{request_timestamp}] /search - Returning {len(processed_results)} papers for '{query}'.")
    return jsonify(processed_results)


# --- Review Generation Routes (Keep as implemented previously) ---

@app.route('/process-and-generate', methods=['POST'])
def handle_process_and_generate_file():
    """ Handles PDF file upload, processing, and review generation. """
    request_timestamp = datetime.now().isoformat()
    app.logger.info(f"[{request_timestamp}] POST /process-and-generate - Received file processing request.")

    if 'pdfFile' not in request.files:
        app.logger.warning(f"[{request_timestamp}] /process-and-generate - No 'pdfFile' part.")
        return jsonify({"error": "No PDF file part in the request"}), 400

    file = request.files['pdfFile']

    if file.filename == '':
        app.logger.warning(f"[{request_timestamp}] /process-and-generate - No selected file.")
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith('.pdf'):
        filename = file.filename
        app.logger.info(f"[{request_timestamp}] /process-and-generate - Processing uploaded file: {filename}")
        try:
            collection_name = pdf_processor.process_pdf_to_chroma(filename, file.stream)
            if not collection_name:
                 app.logger.error(f"[{request_timestamp}] /process-and-generate - Failed to process PDF '{filename}' into ChromaDB.")
                 return jsonify({"error": f"Failed to process content of PDF '{filename}'. It might be empty, corrupted, or unreadable."}), 500

            query_text = f"Summary of the document {Path(filename).stem}"
            app.logger.info(f"[{request_timestamp}] /process-and-generate - Querying collection '{collection_name}' with: '{query_text}'")
            relevant_chunks = pdf_processor.query_chroma_collection(collection_name, query_text, n_results=10)

            app.logger.info(f"[{request_timestamp}] /process-and-generate - Generating review for {filename}...")
            review = pdf_processor.generate_review_from_chunks(relevant_chunks, filename)

            app.logger.info(f"[{request_timestamp}] /process-and-generate - Review generated for {filename}.")
            return jsonify({"review": review})
        except Exception as e:
            app.logger.error(f"[{request_timestamp}] /process-and-generate - Unexpected error for {filename}: {e}", exc_info=True)
            return jsonify({"error": "An internal server error occurred during review generation."}), 500
    else:
        app.logger.warning(f"[{request_timestamp}] /process-and-generate - Invalid file type uploaded: {file.filename}")
        return jsonify({"error": "Invalid file type, please upload a PDF"}), 400


@app.route('/process-url-and-generate', methods=['POST'])
def handle_process_and_generate_url():
    """ Handles PDF download from URL, processing, and review generation. """
    request_timestamp = datetime.now().isoformat()
    app.logger.info(f"[{request_timestamp}] POST /process-url-and-generate - Received URL processing request.")

    if not request.is_json:
        app.logger.warning(f"[{request_timestamp}] /process-url-and-generate - Non-JSON request.")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    pdf_url = data.get('pdfUrl')

    if not pdf_url or not isinstance(pdf_url, str):
        app.logger.warning(f"[{request_timestamp}] /process-url-and-generate - Missing or invalid 'pdfUrl'.")
        return jsonify({"error": "Missing or invalid 'pdfUrl' in request body"}), 400

    app.logger.info(f"[{request_timestamp}] /process-url-and-generate - Processing URL: {pdf_url}")
    try:
        pdf_stream, filename = pdf_processor.download_pdf_from_url(pdf_url)
        if not pdf_stream or not filename:
            app.logger.error(f"[{request_timestamp}] /process-url-and-generate - Failed to download PDF from URL: {pdf_url}")
            return jsonify({"error": f"Failed to download or access PDF from URL: {pdf_url}"}), 502

        app.logger.info(f"[{request_timestamp}] /process-url-and-generate - Processing downloaded file: {filename}")
        collection_name = pdf_processor.process_pdf_to_chroma(filename, pdf_stream)
        pdf_stream.close() # Close the BytesIO stream

        if not collection_name:
            app.logger.error(f"[{request_timestamp}] /process-url-and-generate - Failed to process downloaded PDF '{filename}' into ChromaDB.")
            return jsonify({"error": f"Failed to process content of downloaded PDF '{filename}'."}), 500

        query_text = f"Summary of the document {Path(filename).stem}"
        app.logger.info(f"[{request_timestamp}] /process-url-and-generate - Querying collection '{collection_name}' with: '{query_text}'")
        relevant_chunks = pdf_processor.query_chroma_collection(collection_name, query_text, n_results=10)

        app.logger.info(f"[{request_timestamp}] /process-url-and-generate - Generating review for {filename} from URL...")
        review = pdf_processor.generate_review_from_chunks(relevant_chunks, filename)

        app.logger.info(f"[{request_timestamp}] /process-url-and-generate - Review generated for {filename} from URL.")
        return jsonify({"review": review})
    except Exception as e:
        app.logger.error(f"[{request_timestamp}] /process-url-and-generate - Unexpected error for URL {pdf_url}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during review generation from URL."}), 500


# --- Main Execution ---
if __name__ == '__main__':
    app.logger.info("Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0', port=5000) # Ensure port matches Node.js config