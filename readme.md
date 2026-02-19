# LangChain RAG Project

A Retrieval-Augmented Generation (RAG) system using LangChain, Supabase vector database, and HuggingFace embeddings.

## Project Overview

This project builds a pipeline to:
1. Extract and chunk documents (Markdown & PDF)
2. Generate vector embeddings using sentence-transformers
3. Store embeddings in Supabase vector database
4. Retrieve relevant documents via semantic search
5. Generate answers using LLMs with retrieved context

## Setup

### Prerequisites
- Python 3.13+
- Supabase account with a project
- HuggingFace API (optional, for some models)
- Environment variables: `SUPABASE_URL`, `SUPABASE_KEY`

### Installation

1. **Clone or navigate to the project:**
```bash
cd c:\Projects\langchain-rag
```

2. **Create and activate virtual environment:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file in the project root:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
REPLICATE_API_TOKEN=your_replicate_token (optional)
HUGGINGFACE_API_KEY=your_hf_key (optional)
```

5. **Configure Supabase:**

In your Supabase SQL editor, create the documents table:
```sql
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT NOT NULL,
  metadata JSONB,
  embedding vector(384),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

## Scripts & Workflow

### 1. Extract & Chunk PDFs

**File:** `pdf_to_supabase.py`

Extracts text from PDFs, chunks them, and uploads embeddings to Supabase (with OCR fallback for scanned PDFs).

```powershell
.\.venv\Scripts\python.exe pdf_to_supabase.py "data/PowPay Documentation.pdf" --save-json
```

**Options:**
- `--save-json` — Save chunks to local JSON file
- `--chunk-size` — Set chunk size (default: 500)
- `--chunk-overlap` — Set overlap between chunks (default: 50)
- `--table` — Supabase table name (default: documents)
- `--model` — Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)

**Output:**
- `PowPay Documentation_chunks.json` — 40 chunks extracted from PDF

## Data Processed


### PowPay Documentation (PDF)
- **Source:** `data/PowPay Documentation.pdf`
- **Status:** ✓ Extracted (14 pages) → 40 chunks and uploaded to Supabase
- **Content:** System documentation


## Project Structure

```
c:\Projects\langchain-rag\
├── .venv/                          # Virtual environment
├── .vscode/
│   └── settings.json               # VS Code settings (Python interpreter)
├── data/
│   ├── Template Manager.md         # Chunked & embedded ✓
│   ├── PowPay Documentation.pdf    # Chunked & embedded ✓
│   └── Payroll Sheets.md           # Ready to process
├── chunk_template_manager.py       # Markdown chunking script
├── pdf_to_supabase.py              # PDF→Embedding→Supabase pipeline
├── supabase_test.py                # Supabase connection test
├── populate_database.py            # Legacy Chroma-based loader
├── RAG.py                          # RAG chain implementation (WIP)
├── requirements.txt                # Python dependencies
└── readme.md                       # This file
```

## Database Schema

**Table: documents**

| Column    | Type           | Description                           |
|-----------|----------------|---------------------------------------|
| id        | UUID           | Primary key, auto-generated           |
| content   | TEXT           | Chunk content                         |
| metadata  | JSONB          | Headers, source, page, chunk_id      |
| embedding | vector(384)    | Sentence-transformers embedding      |
| created_at| TIMESTAMP      | Creation timestamp                    |

**Index:** IVFFLAT on embedding column for fast similarity search

## Embeddings Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension:** 384
- **Speed:** Fast, lightweight
- **Use case:** Semantic search, retrieval

## Testing

### Test Supabase Connection
```powershell
.\.venv\Scripts\python.exe supabase_test.py
```

### Verify Embeddings Upload
Query your Supabase table directly:
```sql
SELECT COUNT(*) FROM documents;
SELECT * FROM documents LIMIT 5;
```

## Dependencies

See `requirements.txt` for full list. Key packages:
- `langchain` — LLM orchestration
- `langchain-huggingface` — HuggingFace integrations
- `sentence-transformers` — Embeddings
- `supabase` — Vector database client
- `python-dotenv` — Environment management
- `pypdf` — PDF text extraction

## Troubleshooting

### Import errors in RAG.py
- Imports have changed in newer LangChain versions
- Use `langchain_community` and `langchain_huggingface` instead of legacy imports
- See `pdf_to_supabase.py` for correct import patterns

### PDF text extraction fails
- Scanned/image PDFs require OCR
- Install: `pip install pdf2image pytesseract`
- Download Tesseract binary from https://github.com/tesseract-ocr/tesseract

### Supabase table schema errors
- Ensure table has UUID primary key, not BIGSERIAL
- Embedding column must be `vector(384)` for sentence-transformers model
- Run the SQL schema creation commands above

## License

MIT

## Author

Created with LangChain & Supabase
