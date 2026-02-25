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

load_dotenv()


from completed_rag import invoke_LLM
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
def chat(request: ChatRequest):
    #For Keddish to complete
    #Use invoke_LLM function to process user queries

    return None # Change this

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




