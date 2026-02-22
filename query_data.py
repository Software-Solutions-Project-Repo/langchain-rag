#this file is used to query the RAG system with
# a user-provided query text. It retrieves relevant documents from 
#the Supabase database based on the similarity of their embeddings 
# to the query embedding.

#created by Henry Sylvester on 2026-02-15
import argparse
import os
from dotenv import load_dotenv
from supabase import create_client
from get_embedding_function import get_embedding_function

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)

# This function takes a query text, generates its embedding, and queries the Supabase database for similar documents.
def query_rag(query_text: str):
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    embedding_function = get_embedding_function()

    query_vector = embedding_function.embed_query(query_text)
    response = supabase.rpc(
        "match_documents",
        {"query_embedding": query_vector, "match_count": 5},
    ).execute()
# The response contains the matched documents along with their similarity scores. We print the results in a readable format.
    results = response.data or []
    if not results:
        print("No results found.")
        return
# Print the results with their similarity scores
    for i, row in enumerate(results, 1):
        print(f"\n--- Result {i} (similarity: {row.get('similarity', 'N/A'):.4f}) ---")
        print(row["content"])

    return results


if __name__ == "__main__":
    main()