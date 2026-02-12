import os
import torch
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

model_id = "meta-llama/Llama-3.1-8B"

pipe = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", trust_remote_code=True)
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
