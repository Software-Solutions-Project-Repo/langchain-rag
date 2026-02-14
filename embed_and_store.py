import json
import os
from dotenv import load_dotenv
from supabase import create_client
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

# Initialize embeddings model (using HuggingFace locally instead of Replicate)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load chunks from JSON
with open("chunked_template_manager.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Convert chunks to LangChain Document format
documents = []
for i, chunk in enumerate(chunks):
    doc = Document(
        page_content=chunk["content"],
        metadata={
            "source": "Template Manager",
            "chunk_id": i,
            **chunk["metadata"]  # Add header metadata
        }
    )
    documents.append(doc)

print(f"Loaded {len(documents)} documents from chunks")
print(f"Sample metadata: {documents[0].metadata}\n")

# Create SupabaseVectorStore and add documents
try:
    vector_store = SupabaseVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        client=supabase,
        table_name="documents",
        chunk_size=500
    )
    print(f"✓ Successfully added {len(documents)} vectors to Supabase")
    print(f"✓ Table: documents")
    print(f"✓ Embedding model: sentence-transformers/all-MiniLM-L6-v2")
    
except Exception as e:
    print(f"✗ Error inserting documents: {e}")
    print("\nNote: Make sure your Supabase table 'documents' exists with columns:")
    print("  - id (UUID, primary key)")
    print("  - content (text)")
    print("  - metadata (jsonb)")
    print("  - embedding (vector)")
