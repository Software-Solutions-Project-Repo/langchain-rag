#NOTE: the api key has limited tokens usage and it is valid for 90 days. This starts on 2/22/2026
#Create a .env file for this to work, the api key is AIzaSyBfM7zrx2vv1tPCXPHsSbeaFPjQLIeLTMA, do GOOGLE_API_KEY  = to what u see before for this to work
#If the tokens run out, make your own api key on https://aistudio.google.com/api-keys

from dotenv import load_dotenv
import os

# from langchain_community.llms import HuggingFacePipeline

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
#import google.generativeai as genai




load_dotenv()


#hf_token = os.getenv("HF_TOKEN")

#model_id = "meta-llama/Llama-3.1-8B"



#genai.configure(api_key = os.getenv("GOOGLE_API_KEY" )


llm = ChatGoogleGenerativeAI(model = "gemini-flash-lite-latest", temperature = 0.3, 
                             google_api_key = os.getenv("GOOGLE_API_KEY") )




template = """
You are a payroll assistant bot. Answer the question based on payroll, ignore 
any questions not related to payroll. if you don't know the answer, do not hallucinate it.
Say you don't know the answer. 
For sensitive information such as employee's personal information, say you don't have access to that information.  

Here are your resources {context}

Here is the question to answer: {question}

Answer:


"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm 
context = " This payroll application makes doing payroll easier and more efficient."
while True:
    user_question = input("Ask any question about the payroll service: (Enter 'quit' to exit the program)")

    if user_question.lower() == 'quit':
        print("Thank you for using our payroll assistant chatbot. Have a Good Day!")
        break


    output = chain.invoke({
    "context": context,
    "question": user_question })
    print("Here's the answer to your question: \n" + output.content +"\n")


