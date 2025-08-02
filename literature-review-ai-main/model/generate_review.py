import chromadb
import google.generativeai as genai
import argparse
import os
from pathlib import Path
import sys
from dotenv import load_dotenv # Import the function

# --- Load Environment Variables (Best Practice: Do this early) ---
load_dotenv() # Load variables from the .env file into environment variables

# --- Configuration & Initialization ---

# 1. Load Gemini API Key (Now reads from environment, potentially populated by .env)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GOOGLE_API_KEY not found.")
    print("Please ensure it is set in your .env file or as an environment variable.")
    sys.exit(1)

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    sys.exit(1)

# 2. ChromaDB persistent storage location (MUST match processing.py)
persist_directory = "chroma_db_data"
print(f"Expecting ChromaDB data in: '{persist_directory}'")

# Check if ChromaDB directory exists
if not Path(persist_directory).is_dir():
    print(f"Error: ChromaDB directory '{persist_directory}' not found.")
    print("Please ensure 'processing.py' has run successfully and created the data.")
    sys.exit(1)

# 3. Initialize ChromaDB client
try:
    print("Initializing ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    print(f"ChromaDB client initialized from: {Path(persist_directory).resolve()}")
except Exception as e:
    print(f"Error initializing ChromaDB client: {e}")
    sys.exit(1)

# 4. Gemini Model Selection
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Good balance for speed/cost
print(f"Using Gemini model: {GEMINI_MODEL_NAME}")
try:
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
except Exception as e:
    print(f"Error initializing Gemini model '{GEMINI_MODEL_NAME}': {e}")
    sys.exit(1)


# --- Helper Function (MUST match sanitization in processing.py) ---
def sanitize_collection_name(filename_stem):
    """
    Sanitizes a filename stem to create a valid ChromaDB collection name.
    (This logic MUST match the one used in processing.py)
    """
    # Basic sanitization: replace spaces/dots/hyphens with underscores, make lowercase
    sanitized = "".join(
        c if c.isalnum() or c == '_' or c == '-' else '_'
        for c in filename_stem.lower().replace(" ", "_")
    )
    # Ensure name meets length constraints (e.g., 3-63 chars)
    sanitized = sanitized[:60] # Trim if too long
    # Pad if too short (ChromaDB requires >= 3 chars)
    if len(sanitized) < 3:
        sanitized = f"{sanitized}___"[:3] # Pad with underscores
    return sanitized

# --- Main Review Generation Function ---
def generate_literature_review(collection_name_to_use):
    """
    Retrieves data from ChromaDB, calls Gemini, and prints a literature review.
    """
    try:
        # 1. Get Chroma Collection
        print(f"\nAccessing ChromaDB collection: '{collection_name_to_use}'")
        try:
            collection = chroma_client.get_collection(name=collection_name_to_use)
            print("   Collection found.")
        except ValueError: # Chroma raises ValueError if collection not found via get_collection
             print(f"Error: ChromaDB collection '{collection_name_to_use}' not found.")
             print("   Please ensure the corresponding PDF was processed using processing.py.")
             return # Stop processing for this collection if not found

        # 2. Retrieve ALL documents/chunks for the review
        print("   Retrieving all text chunks...")
        results = collection.get() # Gets all items including documents
        documents = results.get('documents')

        if not documents:
            print(f"Error: No documents found in collection '{collection_name_to_use}'. Cannot generate review.")
            return

        print(f"   Retrieved {len(documents)} text chunks.")

        # Combine chunks into a single context string
        combined_text = "\n\n---\n\n".join(documents)
        print(f"   Combined text length: ~{len(combined_text)} characters.")

        # 3. Prepare Prompt for Gemini API
        prompt = f"""
You are an AI assistant tasked with extracting key information from text chunks of a research paper and presenting it in a structured format.

**Input Text:**
The following text consists of extracted sections from a single research paper:
--- START OF EXTRACTED TEXT ---
{combined_text}
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
        # 4. Call Gemini API
        print(f"\nSending request to Gemini model ({GEMINI_MODEL_NAME})... (This may take a moment)")
        response = model.generate_content(prompt)


        # 5. Print Result
        print("\n--- Generated Literature Review ---")
        try:
            print(response.text)
        except ValueError as e:
             print(f"   Could not extract text from Gemini response: {e}")
             if hasattr(response, 'prompt_feedback'):
                 print(f"   Gemini Prompt Feedback: {response.prompt_feedback}")
             if hasattr(response, 'candidates') and response.candidates:
                 print(f"   Candidate Finish Reason: {response.candidates[0].finish_reason}")
        except Exception as e:
             print(f"   An unexpected error occurred accessing Gemini response text: {e}")

        print("---------------------------------")

    except Exception as e:
        print(f"\nAn unexpected error occurred during review generation for '{collection_name_to_use}':")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Details: {e}")


# --- Main Execution Block ---

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate a literature review for a single processed PDF using its ChromaDB collection and the Gemini API."
    )
    parser.add_argument(
        "--pdf-filename",
        type=str,
        required=True,
        help="The *original* filename of the PDF that was processed (e.g., 'my_research_paper.pdf'). The script derives the collection name from this."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Derive the collection name from the filename
    pdf_path = Path(args.pdf_filename)
    pdf_stem = pdf_path.stem # Filename without extension
    derived_collection_name = sanitize_collection_name(pdf_stem)

    print("-" * 50)
    print(f"Input PDF Filename: {args.pdf_filename}")
    print(f"Derived Collection Name for ChromaDB: {derived_collection_name}")
    print("-" * 50)

    # Call the main review generation function
    # The API key check now happens right after load_dotenv() at the start
    generate_literature_review(derived_collection_name)

    print("\nScript finished.")