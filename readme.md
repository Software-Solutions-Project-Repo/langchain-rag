# LangChain RAG Project

A Retrieval-Augmented Generation (RAG) system using LangChain, Supabase vector database, HuggingFace embeddings, and a FastAPI server for Open WebUI integration.

## Project Overview

This project builds a pipeline to:
1. Extract and chunk PDF documents
2. Generate vector embeddings using sentence-transformers
3. Store embeddings in Supabase vector database across three knowledge stores (documents, question bank, error codes)
4. Retrieve relevant documents via semantic similarity search across all three stores
5. Answer payroll-related questions via a conversational chatbot powered by Google Gemini
6. Serve the chatbot via a FastAPI server with an OpenAI-compatible API for Open WebUI integration

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

In your Supabase SQL editor, create the three knowledge store tables and their RPC functions.

**Documents table** (general documentation chunks):
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

**Question bank table** (Q&A pairs):
```sql
CREATE TABLE qa (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  embedding vector(384),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON qa USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

**Error codes table** (error code Q&A pairs):
```sql
CREATE TABLE error_code_qa (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  error_code TEXT NOT NULL,
  question TEXT,
  answer TEXT,
  embedding vector(384),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON error_code_qa USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

Then create the RPC functions for each table:

**`match_documents`** — searches the documents knowledge store:
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

**`match_qa`** — searches the question bank:
```sql
create or replace function match_qa (
  query_embedding vector(384),
  match_count int default 5,
  filter jsonb default '{}'
) returns table (
  id uuid,
  question text,
  answer text,
  similarity float
)
language sql stable
as $$
  select
    qa.id,
    qa.question,
    qa.answer,
    1 - (qa.embedding <=> query_embedding) as similarity
  from qa
  order by qa.embedding <=> query_embedding
  limit match_count;
$$;
```

**`match_error_code_qa`** — searches the error codes:
```sql
create or replace function match_error_code_qa (
  query_embedding vector(384),
  match_count int default 5,
  filter jsonb default '{}'
) returns table (
  id uuid,
  error_code text,
  question text,
  answer text,
  similarity float
)
language sql stable
as $$
  select
    error_code_qa.id,
    error_code_qa.error_code,
    error_code_qa.question,
    error_code_qa.answer,
    1 - (error_code_qa.embedding <=> query_embedding) as similarity
  from error_code_qa
  order by error_code_qa.embedding <=> query_embedding
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

Embeds a query string and calls three Supabase RPC functions in parallel to retrieve the top 5 most similar results from each knowledge store:
- `match_documents` — searches general documentation chunks
- `match_qa` — searches the question bank (Q&A pairs)
- `match_error_code_qa` — searches error code Q&A pairs

Results from all three stores are combined and returned.

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

An interactive conversational chatbot that combines multi-source Supabase vector search with Google Gemini (`gemini-flash-lite-latest`) to answer payroll-related questions. For each user question it:
1. Queries all three Supabase knowledge stores via `query_rag()`
2. Ranks and selects the top 5 results from each store by similarity score
3. Builds a structured context from question bank entries, error code entries, and document chunks
4. Injects the combined context and chat history into a Gemini prompt
5. Returns a grounded, payroll-focused answer and any embedded images

The bot ignores non-payroll questions and will not hallucinate or expose sensitive employee data.

Can also be imported as a module — `ask_aichatbot_payroll_question(user_question, chat_history)` returns `(answer, images)`.

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

### 4. Run the FastAPI Server

**File:** `main.py`

A FastAPI server that exposes an OpenAI-compatible API, enabling integration with Open WebUI or any OpenAI-compatible client. It wraps `ask_aichatbot_payroll_question()` and supports both streaming and non-streaming responses.

**Endpoints:**
- `GET /` — health check
- `GET /v1/models` — lists available models (returns `payroll-rag`)
- `POST /v1/chat/completions` — accepts chat messages and returns Gemini RAG answers

**Start the server:**
```powershell
python main.py
```

The server runs at `http://localhost:8000`. To connect Open WebUI, add a new model connection pointing to `http://localhost:8000` with model ID `payroll-rag`.

### 5. Run with Docker (Open WebUI)

**Docker Setup:** Run the Open WebUI container using the following command:

```bash
docker run -d -p 3000:8080 -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:dev
```

**Command breakdown:**
- `-d` — Run the container in detached mode (background)
- `-p 3000:8080` — Map port 3000 on your host to port 8080 in the container
- `-v open-webui:/app/backend/data` — Create a persistent volume for application data
- `--name open-webui` — Name the container "open-webui"
- `--restart always` — Automatically restart the container if it stops
- `ghcr.io/open-webui/open-webui:dev` — The Docker image to use (development version)

**Access the UI:**
Once the container is running, access the Open WebUI at `http://localhost:3000`

**Stop the container:**
```bash
docker stop open-webui
```

**Remove the container:**
```bash
docker rm open-webui
```

## Data Processed

### PowPay Documentation (PDF)
- **Source:** `data/PowPay Documentation.pdf`
- **Status:** ✓ Chunked and uploaded to Supabase (`documents` table)
- **Content:** System documentation

## Project Structure

```
c:\Projects\langchain-rag\
├── data/
│   └── PowPay Documentation.pdf    # Source document
├── get_embedding_function.py        # Returns HuggingFace embedding model
├── populate_database.py             # PDF → chunks → embeddings → Supabase
├── query_data.py                    # Query all three Supabase knowledge stores
├── LLM.py                           # Payroll chatbot (Gemini + multi-source RAG)
├── main.py                          # FastAPI server (OpenAI-compatible API)
├── requirements.txt                 # Python dependencies
└── readme.md                        # This file
```

## Database Schema

### Table: `documents`

| Column    | Type        | Description                       |
|-----------|-------------|-----------------------------------|
| id        | UUID        | Primary key, auto-generated       |
| content   | TEXT        | Chunk text content                |
| metadata  | JSONB       | Source file, page number, chunk_id|
| embedding | vector(384) | Sentence-transformers embedding   |
| created_at| TIMESTAMP   | Creation timestamp                |

**RPC:** `match_documents` — cosine similarity search

### Table: `qa`

| Column    | Type        | Description                       |
|-----------|-------------|-----------------------------------|
| id        | UUID        | Primary key, auto-generated       |
| question  | TEXT        | Question text                     |
| answer    | TEXT        | Answer text                       |
| embedding | vector(384) | Embedding of the question         |
| created_at| TIMESTAMP   | Creation timestamp                |

**RPC:** `match_qa` — cosine similarity search

### Table: `error_code_qa`

| Column     | Type        | Description                      |
|------------|-------------|----------------------------------|
| id         | UUID        | Primary key, auto-generated      |
| error_code | TEXT        | Error code identifier            |
| question   | TEXT        | Description / question           |
| answer     | TEXT        | Resolution / answer              |
| embedding  | vector(384) | Embedding for similarity search  |
| created_at | TIMESTAMP   | Creation timestamp               |

**RPC:** `match_error_code_qa` — cosine similarity search

**Index:** IVFFLAT on all embedding columns for fast cosine similarity search

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
- `fastapi` — FastAPI web framework for the API server
- `uvicorn` — ASGI server for FastAPI
- `pydantic` — Data validation for API models

## Troubleshooting

### `column reference "id" is ambiguous` error
The `match_documents` function uses `language plpgsql` which causes a variable scoping conflict. Recreate it using `language sql` as shown in the setup above.

### `403 PERMISSION_DENIED` / API key leaked error
Google automatically revokes keys that are committed to public repositories. Generate a new key at https://aistudio.google.com/api-keys, update `GOOGLE_API_KEY` in your `.env`, and ensure `.env` is in `.gitignore`.

### Supabase table schema errors
- Ensure all tables have a UUID primary key (not BIGSERIAL)
- All embedding columns must be `vector(384)` to match the sentence-transformers model

### PDF text extraction returns 0 characters
- Scanned/image-based PDFs require OCR — install `pdf2image` and `pytesseract`

### Open WebUI shows no response
- Ensure `main.py` is running (`python main.py`) before connecting Open WebUI
- Verify the model ID in Open WebUI matches `payroll-rag`
- Check that `http://localhost:8000/v1/models` returns the model list

## License

MIT

## Authors

Henry Sylvester — Created 2026-02-15  
Jeremiah Clinton 
Keddisha McIntyre
