#created by Jeremiah Clinton on 2026-02-23

import os
import time
import uuid
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from LLM import ask_aichatbot_payroll_question


load_dotenv()


#from completed_rag import invoke_LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

chat_history = []


# ----------- Pydantic Models -----------

# ChatMessage is an object for storing each chat message from Open WebUI
# role can usually be either "user" (Human User) or "assisstant" (AI messages)
# content is the message the user or AI bot sent
class ChatMessage(BaseModel):
    role: str
    content: str

# ChatRequest is a collection of chat messages
class ChatRequest(BaseModel):
    model: str = "payroll-rag"
    messages: list[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None



# ----------- Routes -----------

@app.get("/")
def root():
    return {"Hello": "World"}

# Post request endpoint for sending questions to Open WebUI
@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    try:
        user_msg = next((m.content for m in reversed(request.messages)if m.role == "user"), None)
        if not user_msg:
            return {"error": "No user message found"}

        chat_history.append(f"User:{user_msg}")
        
       
        def event_generator():
            answer = ask_aichatbot_payroll_question(user_msg)
            
            chat_history.append(f"Assistant: {answer}")
            for chunk in answer.split():
                data = {
                    "id": f"chatcml-{int(time.time())}", 
                    "object": "chat.completion.chunk", 
                    "choices": [
                        {
                        "delta": {"content": chunk + " "}, 
                        "index": 0, 
                          
                        }
                        ], 
                    }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

          
        
        if request.stream is False: 
            
           answer = ask_aichatbot_payroll_question(user_msg)

           return {
             "id": f"chatcml-{int(time.time())}", 
             "object": "chat.completion",
             "created": int(time.time()), 
             "model": "payroll-rag", 
             "choices": [
                {
                  "index": 0, 
                  "message": {
                      "role": "assistant", 
                      "content": answer
                    }, 
                   "finish_reason": "stop"
               }
            ], 
        }
        return StreamingResponse(
                event_generator(), 
                media_type = "text/event-stream", 
                )
    except Exception as e:
        print("Error in chat completions route", e)
        return{"error": str(e)}




# This is required by Open WebUI to discover available models.
@app.get("/v1/models")
def list_models(): 
    return {
        "object": "list",
        "data": [
            {
                "id": "payroll-rag", #change payroll-rag to whatever name you gave the "model-id" in Open WebUI
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ]
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)




