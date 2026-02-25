#created by Jeremiah Clinton on 2026-02-23

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from completed_rag import invoke_LLM

app = FastAPI()

# ChatMessage is an object for storing each chat message from Open WebUI
# role can usually be either "user" (Human User) or "assisstant" (AI messages)
# content is the message the user or AI bot sent
class ChatMessage(BaseModel):
    role: str
    content: str

# ChatRequest is a collection of chat messages
# Model = "gemini"
class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]

@app.get("/")
def root():
    return {"Hello": "World"}

# Post request endpoint for sending questions to Open WebUI
@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    #For Keddish to complete
    #Use invoke_LLM function to process user queries

    return {
        "id": "rag-response",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": " " # change this to the variable that you use store user messages in
                },
                "finish_reason": "stop"
            }
        ]
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gemini-rag", #change gemini-rag to whatever name you gave the "model-id" in Open WebUI
                "object": "model",
                "owned_by": "you"
            }
        ]
    }
    




