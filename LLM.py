#NOTE: For the api key , do GOOGLE_API_KEY  = the api key you get from https://aistudio.google.com/api-keys in your env file
 

from dotenv import load_dotenv
from query_data import query_rag
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

For the context of the question, here are some relevant documents that may help you answer the question. Use this information to provide a more accurate and helpful answer to the user's question.

 Here are your resources {context} 

This is the chat History {chat_history} 



Here is the question to answer: {question}

Answer:


"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm 
original_context = " This payroll application makes doing payroll easier and more efficient."

def ask_aichatbot_payroll_question(user_question,chat_history):
    #for blank responses
    if not user_question:
        return "No question provided"
    

    #adding the supabase query results to the context of the prompt to give the LLM more information to work with when answering the user's question.
    #the query results are stored in the variable 'results' and we are adding them t
    context = original_context + "\n" + "\n".join(chat_history)
    results = query_rag(user_question) or []
    

    for r in results:
        content = r.get('content', '')
        if isinstance(content, (list,tuple)):
            content = " ".join(str(x) for x in content)
        context += f"\n -{content}"
    #context += "\n" + "\n".join([f"-{r['content']}" for r in results])
    try:
        output = chain.invoke({
           "context": context, 
            "question": user_question,
            "chat_history": chat_history
       })
        
        
        return output.content
    except Exception as e:
       print("LLM call fail", e)
       return "Cannot Service your request. Sorry"
    
#For testing it in the terminal 
if __name__ == "__main__":
    chat_history = []
    while True:
        
        user_question = input("Ask any question about the payroll service: (Enter 'quit' to exit the program)")

        if user_question.lower() == 'quit':
             print("Thank you for using our payroll assistant chatbot. Have a Good Day!")
             break
        
        chat_history.append(f"User:{user_question}")

        answer = ask_aichatbot_payroll_question(user_question,chat_history)

        chat_history.append(f"Assistant: {answer}")
        print("Here's the answer to your question: \n" + answer +"\n")
