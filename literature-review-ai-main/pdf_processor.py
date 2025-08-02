# pdf_processor.py
import os
import logging
from io import BytesIO
from pathlib import Path

import PyPDF2
import chromadb
import google.generativeai as genai
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# --- Configuration & Initialization ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables (specifically GEMINI_API_KEY)
load_dotenv()

# --- Global Initializations (Load models/clients once) ---
try:
    # Gemini Client
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    LLM_MODEL_NAME = 'gemini-1.5-flash' # Or your preferred model
    llm_model = genai.GenerativeModel(LLM_MODEL_NAME)
    logging.info(f"Gemini client configured successfully for model '{LLM_MODEL_NAME}'.")

    # Sentence Transformer Model
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    logging.info(f"Loading Sentence Transformer model ({EMBEDDING_MODEL_NAME})...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logging.info("Sentence Transformer model loaded successfully.")

    # ChromaDB Client
    CHROMA_PERSIST_DIR = "chroma_db_data"
    logging.info(f"ChromaDB persistent storage location: '{CHROMA_PERSIST_DIR}'")
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    logging.info("Initializing ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    logging.info("ChromaDB client initialized.")

# Handle potential errors during initialization
except ValueError as ve:
    logging.error(f"Configuration Error: {ve}")
    # Depending on desired behavior, you might exit or raise to Flask
    raise ve # Raise to notify Flask app on startup
except ImportError as ie:
     logging.error(f"Import Error: {ie}. Please ensure all dependencies in requirements.txt are installed.")
     raise ie
except Exception as e:
    logging.error(f"Initialization Error: {e}", exc_info=True)
    raise e # Raise other unexpected errors

# --- Core Functions ---

def extract_text_from_pdf(pdf_file_stream) -> str | None:
    """ Extracts text from a PDF file stream. """
    try:
        # Ensure stream is at the beginning
        pdf_file_stream.seek(0)
        reader = PyPDF2.PdfReader(pdf_file_stream)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n" # Add newline between pages
        logging.info(f"Extracted ~{len(full_text)} characters from PDF stream.")
        return full_text if full_text else "" # Return empty string if no text found
    except PyPDF2.errors.PdfReadError as e:
        logging.error(f"PyPDF2 Read Error: {e}. File might be corrupted/encrypted.", exc_info=True)
        return None # Indicate critical read failure
    except Exception as e:
        logging.error(f"Unexpected error extracting PDF text: {e}", exc_info=True)
        return None # Indicate critical failure


def chunk_text(text, chunk_size=500, overlap=50) -> list[str]:
    """ Splits text into overlapping chunks based on word count. """
    if not text: return []
    words = text.split()
    if not words: return []

    chunks = []
    start_index = 0
    while start_index < len(words):
        end_index = min(start_index + chunk_size, len(words)) # Avoid index out of bounds
        chunk_words = words[start_index:end_index]
        chunks.append(" ".join(chunk_words))

        next_start = start_index + chunk_size - overlap
        if next_start <= start_index: # Prevent infinite loop if overlap >= chunk_size
            next_start = start_index + 1
        # Ensure next_start doesn't go beyond the list length
        if next_start >= len(words):
             break
        start_index = next_start

    # Filter out potential empty strings
    filtered_chunks = [chunk for chunk in chunks if chunk and not chunk.isspace()]
    logging.info(f"Split text into {len(filtered_chunks)} chunks.")
    return filtered_chunks


def _sanitize_collection_name(base_name: str) -> str:
    """ Sanitizes a string to be a valid ChromaDB collection name. """
    sanitized = "".join(c if c.isalnum() or c == '_' or c == '-' else '_'
                        for c in base_name.lower().replace(" ", "_"))
    # Enforce length constraints (typically 3-63 chars)
    sanitized = sanitized[:60]
    if len(sanitized) < 3:
        sanitized = f"{sanitized}___"[:3]
    return sanitized


def process_pdf_to_chroma(pdf_filename: str, pdf_content_stream) -> str | None:
    """
    Extracts, chunks, embeds, and stores PDF content in ChromaDB.

    Args:
        pdf_filename: The original name of the PDF file (used for collection name).
        pdf_content_stream: A file-like object (stream) containing the PDF content.

    Returns:
        The sanitized collection name if successful, otherwise None.
    """
    collection_name = _sanitize_collection_name(Path(pdf_filename).stem)
    logging.info(f"Starting processing for '{pdf_filename}' into collection '{collection_name}'")

    extracted_text = extract_text_from_pdf(pdf_content_stream)
    if extracted_text is None: # Check for None explicitly (indicates critical read error)
        logging.error(f"Failed to extract text from {pdf_filename}. Aborting processing.")
        return None
    if not extracted_text: # Handle case where text is empty string
        logging.warning(f"No text content found in {pdf_filename}. Skipping embedding.")
        # Decide: return collection name (as it might exist) or None?
        # Let's return None as nothing was added in this run.
        return None

    chunks = chunk_text(extracted_text)
    if not chunks:
        logging.warning(f"Text from {pdf_filename} resulted in 0 chunks. Skipping embedding.")
        return None

    try:
        logging.info(f"Generating embeddings for {len(chunks)} chunks...")
        # Use the globally loaded model
        embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist() # Keep progress bar off for server logs
        logging.info("Embeddings generated.")

        # Use the globally initialized client
        collection = chroma_client.get_or_create_collection(name=collection_name)
        ids = [f"{collection_name}_chunk_{i}" for i in range(len(chunks))]

        # Add/update data. Chroma's add can often act as upsert if IDs exist,
        # but be aware of potential ID mismatches if chunking changes.
        collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        logging.info(f"Successfully added/updated {len(chunks)} chunks in Chroma collection '{collection_name}'.")
        return collection_name # Return collection name on success
    except Exception as e:
        logging.error(f"Error during embedding or ChromaDB storage for {collection_name}: {e}", exc_info=True)
        return None


def query_chroma_collection(collection_name: str, query_text: str, n_results: int = 5) -> list[str]:
    """ Queries a ChromaDB collection and returns relevant document chunks. """
    if not collection_name or not query_text:
        logging.warning("query_chroma_collection called with empty collection name or query text.")
        return []
    try:
        logging.info(f"Querying collection '{collection_name}' for '{query_text}' (top {n_results} results)...")
        collection = chroma_client.get_collection(name=collection_name) # Get, don't create here

        query_embedding = embedding_model.encode([query_text]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, collection.count()), # Avoid asking for more results than exist
            include=['documents']
        )

        # Extract the documents from the results structure
        retrieved_docs = results['documents'][0] if results and results.get('documents') and results['documents'] else []
        logging.info(f"Retrieved {len(retrieved_docs)} chunks from collection '{collection_name}'.")
        return retrieved_docs

    except chromadb.errors.CollectionNotFoundError:
        logging.warning(f"Collection '{collection_name}' not found during query.")
        return []
    except Exception as e:
        logging.error(f"Error querying Chroma collection '{collection_name}': {e}", exc_info=True)
        return []


def generate_review_from_chunks(chunks: list[str], original_filename="the document") -> str:
    """ Generates a literature review/summary using Gemini based on provided text chunks. """
    if not chunks:
        logging.warning("generate_review_from_chunks called with no chunks.")
        return "Not enough context found in the document to generate a review."

    # --- Create a Prompt for Gemini ---
    context = "\n\n---\n\n".join(chunks)
    # Limit context size if necessary to avoid exceeding LLM token limits
    # max_context_chars = 15000 # Example limit, adjust based on model
    # if len(context) > max_context_chars:
    #     logging.warning(f"Context length ({len(context)}) exceeds limit ({max_context_chars}). Truncating.")
    #     context = context[:max_context_chars]

    # Improved prompt asking for specific aspects
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

    logging.info(f"Generating review for '{original_filename}' using {len(chunks)} chunks...")
    try:
        # Use the globally configured Gemini model
        response = llm_model.generate_content(prompt)
        # Add more robust response handling if needed (check finish reasons, safety ratings etc.)
        review_text = response.text
        logging.info("Review generated successfully via Gemini.")
        return review_text
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}", exc_info=True)
        return "Error: Could not generate the review due to an issue with the AI service."


def download_pdf_from_url(pdf_url: str, timeout: int = 30) -> tuple[BytesIO | None, str | None]:
    """
    Downloads a PDF from a URL into an in-memory BytesIO object.

    Args:
        pdf_url: The URL of the PDF.
        timeout: Request timeout in seconds.

    Returns:
        A tuple containing (BytesIO stream, filename) on success, or (None, None) on failure.
    """
    try:
        logging.info(f"Attempting to download PDF from URL: {pdf_url}")
        headers = {'User-Agent': 'Mozilla/5.0'} # Some sites block default requests User-Agent
        response = requests.get(pdf_url, stream=True, timeout=timeout, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        # Check content type if possible
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' not in content_type:
            logging.warning(f"URL content-type ('{content_type}') is not 'application/pdf'. Attempting to process anyway.")
            # Decide whether to proceed or return error

        # Get filename from headers or URL
        content_disposition = response.headers.get('content-disposition')
        filename = None
        if content_disposition:
            # Extract filename from Content-Disposition header (more complex parsing may be needed)
            parts = content_disposition.split('filename=')
            if len(parts) > 1:
                filename = parts[1].strip('" ')
        if not filename:
            # Fallback to deriving filename from URL path
            try:
                 filename = Path(requests.utils.urlparse(pdf_url).path).name
            except Exception:
                 filename = "downloaded_paper.pdf" # Generic fallback

        # Ensure filename has .pdf extension
        if not filename.lower().endswith('.pdf'):
             filename += ".pdf"


        pdf_stream = BytesIO(response.content)
        logging.info(f"Successfully downloaded PDF '{filename}' into memory stream.")
        return pdf_stream, filename

    except requests.exceptions.Timeout:
        logging.error(f"Timeout error downloading PDF from {pdf_url}", exc_info=True)
        return None, None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading PDF from {pdf_url}: {e}", exc_info=True)
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error during PDF download from {pdf_url}: {e}", exc_info=True)
        return None, None