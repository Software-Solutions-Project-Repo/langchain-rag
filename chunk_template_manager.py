from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import json

# Read the Template Manager markdown file
with open("data/Template Manager.md", "r", encoding="utf-8") as f:
    markdown_content = f.read()

# Split by markdown headers to preserve section structure
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_line=False
)

# First split by markdown headers
md_header_splits = markdown_splitter.split_text(markdown_content)

# Then apply recursive character splitting for finer granularity
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Further split the sections if they're too long
chunks = []
for doc in md_header_splits:
    split_chunks = text_splitter.split_text(doc.page_content)
    for chunk_text in split_chunks:
        chunks.append({
            "content": chunk_text,
            "metadata": doc.metadata
        })

# Display and save chunks
print(f"Total chunks: {len(chunks)}\n")
print("=" * 80)

for i, chunk in enumerate(chunks, 1):
    print(f"\n[Chunk {i}]")
    print(f"Metadata: {chunk['metadata']}")
    print(f"Content:\n{chunk['content']}")
    print("-" * 80)

# Save to JSON for later use
with open("chunked_template_manager.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(chunks)} chunks to chunked_template_manager.json")
