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
You are a payroll assistant bot. Answer the question based on payroll, error codes and from the question bank, ignore 
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
    
    results = query_rag(user_question) or []
    #adding the supabase query results to the context of the prompt to give the LLM more information to work with when answering the user's question.

     #Breaking the query up into the various sections such as question bank, documents and error code so that the llm can filter the most relevant answer 
    doc_results = [r for r in results if r.get('content') and not r.get('question')]
    qa_results =[r for r in results if r.get('question') and r.get('answer')]
    error_results = [r for r in results if r.get('error_code')]

   #this finds the similar answer 
    def get_similarity(r):
        similarity =  r.get('similarity')
        return similarity if similarity is not None else 0.0

    doc_results.sort(key=get_similarity , reverse = True)
    qa_results.sort(key=get_similarity , reverse = True)
    error_results.sort(key=get_similarity, reverse = True)

    selected = doc_results[:3] +qa_results[:3] + error_results[:3]
    
    context = original_context + "\n\n"

    qa_selected =[r for r in selected if r.get('question') and r.get('answer')]
    if qa_selected:
       context += "--From the question Bank--"
       for r in qa_selected:
          context += f"Q:{r['question']}\nA: {r['answer']}\n\n"
    

    error_selected =[r for r in selected if r.get('error_code')]
    if error_selected:
       context += "--From the Error--"
       for r in error_selected:
          context += f"Error:{r['error_code']}: {r.get('question', '')} -{r.get('answer','')}\n\n"

    
    doc_selected =[r for r in selected if r.get('content') and not r.get('question')]
    if doc_selected:
       context += "--From the document--"
       for r in doc_selected:
          context += f"Q:{r['content']}\n\n"

  

    if context.strip() == " ":
        context += "\nNo relevant information found."
    try:
        output = chain.invoke({
           "context": context, 
            "question": user_question,
            "chat_history": "\n" .join(chat_history)
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
