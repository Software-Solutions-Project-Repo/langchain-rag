#created by Jeremiah Clinton on 2026-02-23

import os
from dotenv import load_dotenv
import argparse
from supabase import create_client
from query_data import query_rag
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

llm = ChatGoogleGenerativeAI(model = "gemini-flash-lite-latest", temperature = 0.3, 
                             google_api_key = os.getenv("GEMINI_API_KEY") ) 
# Change "GEMINI_API_KEY" to the name of your environment variable with your Google Gemini API key.


# So that the LLM doesn't hallucinate or give unrelated responses
template = """
You are a payroll assistant bot. Answer the question based on payroll, ignore 
any questions not related to payroll. if you don't know the answer, do not hallucinate it.
Say you don't know the answer. 
For sensitive information such as employee's personal information, say you don't have access to that information. 

For the context of the question, here are some relevant documents that may help you answer the question. Use this information to provide a more accurate and helpful answer to the user's question.

Here are your resources {context} 

Here is the question to answer: {question}

Answer:


"""

# Function for calling the LLM
def invoke_LLM():
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm 
    context = " This payroll application makes doing payroll easier and more efficient."
    while True:
        user_question = input("Ask any question about the payroll service: (Enter 'quit' to exit the program)")

        if user_question.lower() == 'quit':
            print("Thank you for using our payroll assistant chatbot. Have a Good Day!")
            break
        #adding the supabase query results to the context of the prompt to give the LLM more information to work with when answering the user's question.
        #the query results are stored in the variable 'results' and we are adding them to
        results = query_rag(user_question)
        context += "\n".join([f"- {result['content']}" for result in results])

        output = chain.invoke({
        "context": context,
        "question": user_question })
        print("Here's the answer to your question: \n" + output.content +"\n")


def main():
    invoke_LLM()
    

    

if __name__ == "__main__":
    main()
    

    



    