import os
from dotenv import load_dotenv
import argparse
from supabase import create_client

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
# from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# Initialize Supabase client
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

# Embeddniig model used to turn user queries into vectors
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to existing Supabase vector store
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents"
)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or gemini-1.5-pro
    temperature=0.2
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Use ONLY the provided context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""")

output_parser = StrOutputParser()


def ask_question(question: str, k: int = 5):
    print(f"\n🔎 Searching for relevant documents...")

    # 1️⃣ Retrieve top-k similar documents
    docs = vector_store.similarity_search(question, k=k)

    if not docs:
        return "No relevant documents found."

    # 2️⃣ Combine context
    context = "\n\n".join([doc.page_content for doc in docs])

    print(f"✓ Retrieved {len(docs)} documents")

    # 3️⃣ Create RAG chain
    chain = prompt | llm | output_parser

    # 4️⃣ Run LLM with context
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response


if __name__ == "__main__":
    while True:
        user_question = input("\nAsk a question (or type 'exit'): ")
        if user_question.lower() == "exit":
            break

        answer = ask_question(user_question)
        print("\n💬 Answer:")
        print(answer)