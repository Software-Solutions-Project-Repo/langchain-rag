#this code is used to populate the database with the pdf file. It will chunk the pdf file and 
# upload the chunks to the database. It also has an option to save the chunks to a json file locally.
# The code uses OCR as a fallback if the pdf file does not contain extractable text. The code uses HuggingFaceEmbeddings 
# to create embeddings for the chunks and uploads them to SupabaseVectorStore.
#created by Henry Sylvester on 2026-02-15

import argparse
import os
import json
from dotenv import load_dotenv
from supabase import create_client

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    parser = argparse.ArgumentParser(description="Chunk a PDF and upload embeddings to Supabase")
    parser.add_argument("pdf", help="Path to PDF file (relative to workspace)")
    parser.add_argument("--table", default="documents", help="Supabase table name to store vectors")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    load_dotenv()
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

    # Try extracting text via PyPDFLoader
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
    #checking to see if there is anything to embed before creating embeddings and uploading to the database
    if len(chunks) == 0:
        print("No chunks to embed — exiting.")
        return

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=args.model)

    # Convert chunks to Documents for SupabaseVectorStore
    docs_for_store = []
    for i, c in enumerate(chunks):
        if isinstance(c, Document):
            docs_for_store.append(c)
        else:
            docs_for_store.append(Document(page_content=c.page_content, metadata={"source": pdf_path, "chunk_id": i, **(c.metadata or {})}))

    # Upload to Supabase
    try:
        SupabaseVectorStore.from_documents(
            documents=docs_for_store,
            embedding=embeddings,
            client=supabase,
            table_name=args.table,
        )
        print(f"✓ Uploaded {len(docs_for_store)} vectors to Supabase table '{args.table}'")
    except Exception as e:
        print("✗ Error inserting documents:", e)
        print("Ensure the Supabase table exists with columns: id (UUID), content (text), metadata (jsonb), embedding (vector).")


if __name__ == "__main__":
    main()
