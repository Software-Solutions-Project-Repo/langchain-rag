# LangChain RAG Project

A Retrieval-Augmented Generation (RAG) system using LangChain, Supabase vector database, and HuggingFace embeddings.

## Project Overview

This project builds a pipeline to:
1. Extract and chunk PDF documents
2. Generate vector embeddings using sentence-transformers
3. Store embeddings in Supabase vector database
4. Retrieve relevant documents via semantic similarity search

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
```

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
- `sentence-transformers` — Local embedding model
- `supabase` — Supabase client
- `python-dotenv` — Environment variable management
- `pypdf` — PDF text extraction

## Troubleshooting

### `column reference "id" is ambiguous` error
The `match_documents` function uses `language plpgsql` which causes a variable scoping conflict. Recreate it using `language sql` as shown in the setup above.

### Supabase table schema errors
- Ensure the table has a UUID primary key (not BIGSERIAL)
- The embedding column must be `vector(384)` to match the sentence-transformers model

### PDF text extraction returns 0 characters
- Scanned/image-based PDFs require OCR — install `pdf2image` and `pytesseract`

## License

MIT

## Author

Henry Sylvester — Created 2026-02-15
