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


def ocr_pdf_to_documents(pdf_path):
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception as e:
        print("OCR dependencies missing: install `pdf2image`, `pytesseract`, and the Tesseract binary.")
        raise

    images = convert_from_path(pdf_path)
    docs = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        docs.append(Document(page_content=text, metadata={"source": pdf_path, "page": i + 1}))
    return docs


def main():
    parser = argparse.ArgumentParser(description="Chunk a PDF and upload embeddings to Supabase")
    parser.add_argument("pdf", help="Path to PDF file (relative to workspace)")
    parser.add_argument("--table", default="documents", help="Supabase table name to store vectors")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--save-json", action="store_true", help="Save chunks to JSON locally")
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

    # If pages are empty, attempt OCR fallback
    total_text_chars = sum(len(d.page_content.strip()) for d in documents)
    if total_text_chars == 0:
        print("No extractable text found in PDF pages — attempting OCR fallback...")
        try:
            documents = ocr_pdf_to_documents(pdf_path)
        except Exception:
            print("OCR failed or dependencies missing. Export text externally or install OCR tools.")
            return

    print(f"Loaded {len(documents)} pages. Total characters: {sum(len(d.page_content) for d in documents)}")

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Produced {len(chunks)} chunks")

    if args.save_json:
        out = [{"content": c.page_content, "metadata": c.metadata} for c in chunks]
        out_path = os.path.splitext(os.path.basename(pdf_path))[0] + "_chunks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved chunks to {out_path}")

    if len(chunks) == 0:
        print("No chunks to embed — exiting.")
        return

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=args.model)

    # Convert chunks to Documents (if not already)
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
