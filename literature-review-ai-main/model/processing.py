import PyPDF2
from io import BytesIO
from sentence_transformers import SentenceTransformer
import chromadb
import os
import argparse # For handling command-line arguments
from pathlib import Path # For easier path manipulation

# --- Configuration & Initialization (Best Practice: Near the top) ---

# Load the Sentence Transformer model (consider loading only if needed, but often fine globally)
try:
    # Using a specific cache folder for models can be good practice
    # model_cache_folder = os.path.join(Path.home(), ".cache", "sentence_transformers")
    # os.makedirs(model_cache_folder, exist_ok=True)
    # model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=model_cache_folder)

    # Simpler default cache usage:
    print("Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}")
    print("Please ensure sentence-transformers and its dependencies (like PyTorch/TensorFlow) are installed correctly.")
    exit(1) # Exit if model loading fails

# ChromaDB persistent storage location (configurable)
persist_directory = "chroma_db_data"
print(f"ChromaDB persistent storage location set to: '{persist_directory}'")

# Ensure the directory exists
try:
    os.makedirs(persist_directory, exist_ok=True)
except OSError as e:
    print(f"Error creating ChromaDB directory '{persist_directory}': {e}")
    exit(1)

# Initialize ChromaDB client
try:
    print("Initializing ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    print(f"ChromaDB client initialized. Data will be stored in: {Path(persist_directory).resolve()}")
except Exception as e:
    print(f"Error initializing ChromaDB client: {e}")
    exit(1)

# --- Core Functions (with Docstrings - Best Practice) ---

def extract_text_from_pdf(pdf_file_object):
    """
    Extracts text from a PDF file object.

    Args:
        pdf_file_object: A file-like object opened in binary read mode ('rb').

    Returns:
        A string containing the extracted text, or an empty string if no text
        could be extracted or in case of an error. Returns None on read error.
    """
    try:
        reader = PyPDF2.PdfReader(pdf_file_object)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n" # Add newline between pages
        return full_text
    except PyPDF2.errors.PdfReadError as e:
         print(f"   Error reading PDF structure: {e}. File might be corrupted or encrypted.")
         return None
    except Exception as e:
        print(f"   An unexpected error occurred during text extraction: {e}")
        return "" # Return empty string for other unexpected errors

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks based on word count.

    Args:
        text (str): The input text.
        chunk_size (int): The target number of words per chunk.
        overlap (int): The number of words to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []
    words = text.split()
    if not words:
        return []

    chunks = []
    start_index = 0
    while start_index < len(words):
        end_index = start_index + chunk_size
        chunk_words = words[start_index:end_index]
        chunks.append(" ".join(chunk_words))
        # Move start index for the next chunk, considering overlap
        next_start = start_index + chunk_size - overlap
        # If overlap is too large or chunk size is small, prevent infinite loops
        if next_start <= start_index:
            next_start = start_index + 1 # Move by at least one word
        start_index = next_start

    # Filter out potential empty strings if any edge cases create them
    return [chunk for chunk in chunks if chunk]


def process_and_store_pdf(pdf_path_str):
    """
    Processes a single PDF file: extracts text, chunks, embeds, and stores in ChromaDB.

    Args:
        pdf_path_str (str): The path to the PDF file.
    """
    pdf_path = Path(pdf_path_str)

    # --- Input Validation (Best Practice) ---
    if not pdf_path.is_file():
        print(f"Error: PDF file not found at '{pdf_path}'")
        return
    if not pdf_path.name.lower().endswith(".pdf"):
        print(f"Warning: File '{pdf_path.name}' does not have a .pdf extension.")
        # Decide if you want to proceed or return here

    # --- Determine Collection Name (Best Practice: Sanitize) ---
    # Use filename without extension as base for collection name
    collection_name_base = pdf_path.stem
    # Basic sanitization: replace spaces/dots/hyphens with underscores, make lowercase
    # ChromaDB has validation rules for collection names (e.g., length, allowed chars)
    # Check ChromaDB docs for specifics, this is a reasonable starting point:
    sanitized_collection_name = "".join(
        c if c.isalnum() or c == '_' or c == '-' else '_'
        for c in collection_name_base.lower().replace(" ", "_")
    )
    # Ensure name meets length constraints (e.g., 3-63 chars)
    sanitized_collection_name = sanitized_collection_name[:60] # Trim if too long
    if len(sanitized_collection_name) < 3: # Pad if too short
        sanitized_collection_name = f"{sanitized_collection_name}___"[:3]


    print("-" * 50)
    print(f"Processing PDF: '{pdf_path.name}'")
    print(f"Using ChromaDB collection name: '{sanitized_collection_name}'")

    try:
        # --- Step 1: Extract Text ---
        print("\nStep 1: Extracting text...")
        with open(pdf_path, "rb") as f: # Read in binary mode ('rb') is required by PyPDF2
            extracted_text = extract_text_from_pdf(f)

        if extracted_text is None: # Indicates a read error from extract_text_from_pdf
            print("Failed to process PDF due to read error.")
            return
        if not extracted_text:
            print("Warning: No text could be extracted. Skipping embedding and storage.")
            return
        print(f"   Extracted ~{len(extracted_text)} characters.")

        # --- Step 2: Chunk Text ---
        print("\nStep 2: Chunking text...")
        chunks = chunk_text(extracted_text, chunk_size=500, overlap=50) # Use defined function
        if not chunks:
            print("Warning: Text could not be chunked (perhaps it was too short?). Skipping.")
            return
        print(f"   Created {len(chunks)} chunks.")

        # --- Step 3: Embed Chunks ---
        print("\nStep 3: Generating embeddings...")
        embeddings = model.encode(chunks, show_progress_bar=True).tolist() # Show progress bar
        print(f"   Generated {len(embeddings)} embeddings.")

        # --- Step 4: Store in ChromaDB ---
        print("\nStep 4: Storing in ChromaDB...")
        # Get or create the collection
        collection = chroma_client.get_or_create_collection(name=sanitized_collection_name)

        # Create unique IDs for each chunk (important for potential updates/deletions)
        ids = [f"{sanitized_collection_name}_chunk_{i}" for i in range(len(chunks))]

        # Add data to the collection
        # Note: ChromaDB might embed automatically if embeddings=None and embedding_function is set
        # But since we pre-calculated, we provide them directly.
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        print(f"   Successfully added/updated {len(chunks)} chunks in collection '{sanitized_collection_name}'.")

    except FileNotFoundError:
        # This case should be caught by the initial check, but good practice to keep
        print(f"Error: File disappeared during processing: '{pdf_path}'")
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing '{pdf_path.name}':")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Details: {e}")
        # Consider adding more specific error handling for ChromaDB operations if needed

    finally:
        print("-" * 50)


# --- Main Execution Block (Best Practice: Guard with if __name__ == "__main__":) ---

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF, chunk it, generate embeddings, and store in ChromaDB."
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        required=True,
        help="Path to the PDF file that needs to be processed."
    )
    # Potential future arguments:
    # parser.add_argument("--collection-name", type=str, help="Override default collection name (derived from filename)")
    # parser.add_argument("--chunk-size", type=int, default=500, help="Word count for text chunks")
    # parser.add_argument("--overlap", type=int, default=50, help="Word overlap between chunks")
    # parser.add_argument("--persist-dir", type=str, default=persist_directory, help="Override ChromaDB storage directory")

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Call the main processing function with the provided PDF path
    process_and_store_pdf(args.pdf_path)

    print("\nScript finished.")