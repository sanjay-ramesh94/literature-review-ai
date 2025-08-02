# Literature Review AI - Hackathon Project


## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Velavan5/literature-review-ai.git
    cd literature-review-ai
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Create `.env` file:**
    If you have a Semantic Scholar API key, create a `.env` file in the project root and add:
    ```
    SEMANTIC_SCHOLAR_API_KEY=YOUR_KEY_HERE
    ```

5.  **Run the Flask development server:**
    ```bash
    python app.py
    ```
    The server will start, usually at `http://127.0.0.1:5000/` .
## Usage

Send a POST request to the `/search` endpoint with a JSON body:

**Example using `curl`:**

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"query": "large language models efficiency", "limit": 5}' \
     [http://127.0.0.1:5000/search](http://127.0.0.1:5000/search)
