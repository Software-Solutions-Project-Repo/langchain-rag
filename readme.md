# LangChain RAG Project

A Retrieval-Augmented Generation (RAG) system using LangChain, Supabase vector database, and HuggingFace embeddings.

## Project Overview

This project builds a pipeline to:
1. Extract and chunk PDF documents
2. Generate vector embeddings using sentence-transformers
3. Store embeddings in Supabase vector database
4. Retrieve relevant documents via semantic similarity search
5. Answer payroll-related questions via a conversational chatbot powered by Google Gemini

## Setup

### Prerequisites
- Python 3.13+
- Supabase account with a project
- Environment variables: `SUPABASE_URL`, `SUPABASE_KEY`

### Installation

1. **Clone or navigate to the project:**
```bash
cd c:\Projects\langchain-rag
```

2. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the project root:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GOOGLE_API_KEY=your_gemini_api_key
```

Get a Gemini API key at https://aistudio.google.com/api-keys. The key is valid for 90 days with a limited token quota.

> **Important:** Never commit `.env` to git. Add it to `.gitignore` to prevent your API key from being leaked.

4. **Configure Supabase:**

In your Supabase SQL editor, create the documents table and similarity search function:
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

Then create the `match_documents` RPC function used for vector similarity search:
```sql
create or replace function match_documents (
  query_embedding vector(384),
  match_count int default 5,
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language sql stable
as $$
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where documents.metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
$$;
```

## Scripts & Workflow

### 1. Populate the Database

**File:** `populate_database.py`

Loads a PDF, chunks it using `RecursiveCharacterTextSplitter`, generates embeddings with HuggingFace, and uploads them to Supabase.

```powershell
python populate_database.py "data/PowPay Documentation.pdf"
```

**Options:**
- `--table` — Supabase table name (default: `documents`)
- `--chunk-size` — Size of each chunk in characters (default: 500)
- `--chunk-overlap` — Overlap between chunks (default: 50)
- `--model` — Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)

### 2. Query the Database

**File:** `query_data.py`

Embeds a query string and calls the `match_documents` RPC to retrieve the top 5 most similar chunks from Supabase, printing each result with its similarity score.

```powershell
python query_data.py "how to create an employee?"
```

**Output example:**
```
--- Result 1 (similarity: 0.8812) ---
To create an employee, navigate to...

--- Result 2 (similarity: 0.8540) ---
...
```

### 3. Run the Payroll Chatbot

**File:** `LLM.py`

An interactive conversational chatbot that combines Supabase vector search with Google Gemini to answer payroll-related questions. For each user question it:
1. Queries Supabase for the most relevant document chunks via `query_rag()`
2. Injects those results as context into a Gemini prompt
3. Returns a grounded, payroll-focused answer

The bot ignores non-payroll questions and will not hallucinate or expose sensitive employee data.

```powershell
python LLM.py
```

**Example session:**
```
Ask any question about the payroll service: (Enter 'quit' to exit the program)
> how do I create an employee?
Here's the answer to your question:
To create an employee, navigate to...
```

Enter `quit` to exit.

## Data Processed

### PowPay Documentation (PDF)
- **Source:** `data/PowPay Documentation.pdf`
- **Status:** ✓ Chunked and uploaded to Supabase
- **Content:** System documentation

## Project Structure

```
c:\Projects\langchain-rag\
├── data/
│   └── PowPay Documentation.pdf    # Source document
├── get_embedding_function.py        # Returns HuggingFace embedding model
├── populate_database.py             # PDF → chunks → embeddings → Supabase
├── query_data.py                    # Query Supabase by semantic similarity
├── LLM.py                           # Payroll chatbot (Gemini + RAG)
├── requirements.txt                 # Python dependencies
└── readme.md                        # This file
```

## Database Schema

**Table: documents**

| Column    | Type        | Description                       |
|-----------|-------------|-----------------------------------|
| id        | UUID        | Primary key, auto-generated       |
| content   | TEXT        | Chunk text content                |
| metadata  | JSONB       | Source file, page number, chunk_id|
| embedding | vector(384) | Sentence-transformers embedding   |
| created_at| TIMESTAMP   | Creation timestamp                |

**Index:** IVFFLAT on embedding column for fast cosine similarity search

## Embeddings Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension:** 384
- **Speed:** Fast, lightweight
- **Use case:** Semantic search and retrieval

## Dependencies

See `requirements.txt` for full list. Key packages:
- `langchain-huggingface` — HuggingFace embedding integration
- `langchain-community` — Document loaders and text splitters
- `langchain-google-genai` — Google Gemini LLM integration
- `sentence-transformers` — Local embedding model
- `supabase` — Supabase client
- `python-dotenv` — Environment variable management
- `pypdf` — PDF text extraction

## Troubleshooting

### `column reference "id" is ambiguous` error
The `match_documents` function uses `language plpgsql` which causes a variable scoping conflict. Recreate it using `language sql` as shown in the setup above.

### `403 PERMISSION_DENIED` / API key leaked error
Google automatically revokes keys that are committed to public repositories. Generate a new key at https://aistudio.google.com/api-keys, update `GOOGLE_API_KEY` in your `.env`, and ensure `.env` is in `.gitignore`.

### Supabase table schema errors
- Ensure the table has a UUID primary key (not BIGSERIAL)
- The embedding column must be `vector(384)` to match the sentence-transformers model

### PDF text extraction returns 0 characters
- Scanned/image-based PDFs require OCR — install `pdf2image` and `pytesseract`

## License

MIT

## Author

Henry Sylvester — Created 2026-02-15
