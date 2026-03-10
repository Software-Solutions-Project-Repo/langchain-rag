#this code is used to populate the database with the pdf file. It will chunk the pdf file and 
# upload the chunks to the database. It also has an option to save the chunks to a json file locally.
# The code uses OCR as a fallback if the pdf file does not contain extractable text. The code uses HuggingFaceEmbeddings 
# to create embeddings for the chunks and uploads them to SupabaseVectorStore.
# Documents are stored in a system-specific table and registered in the master 'document_registry' table.
#created by Henry Sylvester on 2026-02-15

import argparse
import os
import re
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client
import psycopg2

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

MASTER_TABLE = "document_registry"


def sanitize_table_name(name: str) -> str:
    """Convert a system name into a safe Supabase table name (lowercase, underscores, no leading digit)."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)  # replace non-alphanumeric runs with _
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "sys_" + name
    return name or "documents"


def ensure_tables_exist(db_url: str, table_name: str, embedding_dim: int = 384):
    """Create the system vector table and document_registry if they don't already exist."""
    # table_name is already sanitized to [a-z0-9_] so safe to interpolate
    sql_system = f"""
        create extension if not exists vector;
        create table if not exists {table_name} (
            id        uuid primary key default gen_random_uuid(),
            content   text,
            metadata  jsonb,
            embedding vector({embedding_dim})
        );
    """
    
def register_document(supabase, system_name: str, table_name: str, document_name: str, chunks: int):
    """Insert a record into the master document_registry table."""
    record = {
        "system_name": system_name,
        "table_name": table_name,
        "document_name": document_name,
        "chunks_uploaded": chunks,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        supabase.table(MASTER_TABLE).insert(record).execute()
        print(f"✓ Registered in master table '{MASTER_TABLE}'")
    except Exception as e:
        print(f"⚠ Could not write to master table '{MASTER_TABLE}': {e}")
        print(
            f"  Ensure the table exists with columns: "
            "id (uuid, default gen_random_uuid()), system_name (text), table_name (text), "
            "document_name (text), chunks_uploaded (int4), uploaded_at (timestamptz)."
        )


def main():
    parser = argparse.ArgumentParser(description="Chunk a PDF and upload embeddings to Supabase")
    parser.add_argument("pdf", help="Path to PDF file (relative to workspace)")
    parser.add_argument(
        "--system",
        required=True,
        help="Name of the system this document belongs to (e.g. 'HR System'). "
             "Used to derive the target table name.",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Override the Supabase table name. Defaults to a sanitized version of --system.",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-dim", type=int, default=384,
                        help="Embedding vector dimension (default 384 for all-MiniLM-L6-v2).")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # Derive table name from system name unless explicitly overridden
    table_name = args.table if args.table else sanitize_table_name(args.system)
    document_name = os.path.basename(pdf_path)

    print(f"System  : {args.system}")
    print(f"Table   : {table_name}")
    print(f"Document: {document_name}")

    load_dotenv()
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

    # Auto-create tables using the direct PostgreSQL connection string.
   
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        ensure_tables_exist(db_url, table_name, args.embedding_dim)
    else:
        print("ℹ No DATABASE_URL found in .env — skipping auto table creation.")
        print("  Add DATABASE_URL=postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres")

    # Extract text via PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages. Total characters: {sum(len(d.page_content) for d in documents)}")

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Produced {len(chunks)} chunks")

    # Check there is anything to embed before creating embeddings and uploading to the database
    if len(chunks) == 0:
        print("No chunks to embed — exiting.")
        return

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=args.model)

    # Attach system_name to every chunk's metadata so it is queryable later
    docs_for_store = []
    for i, c in enumerate(chunks):
        meta = {
            "source": pdf_path,
            "chunk_id": i,
            "system_name": args.system,
            "document_name": document_name,
            **(c.metadata if isinstance(c, Document) else (c.metadata or {})),
        }
        docs_for_store.append(Document(page_content=c.page_content, metadata=meta))

    # Upload vectors to the system-specific table
    try:
        SupabaseVectorStore.from_documents(
            documents=docs_for_store,
            embedding=embeddings,
            client=supabase,
            table_name=table_name,
        )
        print(f"✓ Uploaded {len(docs_for_store)} vectors to Supabase table '{table_name}'")
    except Exception as e:
        print("✗ Error inserting documents:", e)
        print(
            f"  Ensure the table '{table_name}' exists with columns: "
            "id (uuid), content (text), metadata (jsonb), embedding (vector)."
        )
        return

    # Record this upload in the master registry
    register_document(supabase, args.system, table_name, document_name, len(docs_for_store))


if __name__ == "__main__":
    main()
