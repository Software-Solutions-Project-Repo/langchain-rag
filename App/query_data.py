#this file is used to query the RAG system with
# a user-provided query text. It retrieves relevant documents from 
#the Supabase database based on the similarity of their embeddings 
# to the query embedding.

#created by Henry Sylvester on 2026-02-15
import argparse
import os
from dotenv import load_dotenv
from supabase import create_client
from App.get_embedding_function import get_embedding_function

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

    try:

       query_vector = embedding_function.embed_query(query_text)
       response = supabase.rpc(
           "match_documents",
           {"query_embedding": query_vector, "match_count": 5},
         ).execute()
    except Exception as e:
           print(f"Error calling match_documents: {e}")  

    #This queries the question bank 
    try:
        question_response = supabase.rpc(
        "match_qa",
        {"query_embedding": query_vector, "match_count": 5, "filter" :{}
         },
    ).execute()
       
    except Exception as e:
        print(f"Error calling match_qa: {e}")
    try:
        if not question_response.data:
            print("empty, retrying without filter")
            question_response = supabase.rpc(
            "match_qa",
            {"query_embedding": query_vector, "match_count": 5, "filter" :{}
            },
            ).execute()
      
    except Exception as e:
           print(f"Error calling match_qa: {e}")  
            #This queries the Error code 

    try:
        error_code_response = supabase.rpc(
            "match_error_code_qa",
             {"query_embedding": query_vector, "match_count": 5, "filter":{}},
             ).execute()
        
    except Exception as e:
           print(f"Error calling match_error_code_qa: {e}")  
    
    try:
        if not error_code_response.data:
            print("empty, retrying without filter")
            error_code_response = supabase.rpc(
            "match_error_code_qa",
            {"query_embedding": query_vector, "match_count": 5},
            ).execute()
       
    except Exception as e:
           print(f"Error calling match_error_code_qa: {e}")  
# The response contains the matched documents along with their similarity scores. We print the results in a readable format.
    results = (response.data or [] ) +(question_response.data or [])+(error_code_response.data or [])
    if not results:
        print("No results found.")
        return
    
    return results


if __name__ == "__main__":
    main()