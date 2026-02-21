import argparse
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import login
from transformers import pipeline
from google import genai
from get_embedding_function import get_embedding_function

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context. If the answer is not in the context, say "I don't know."

Context:
{context}

---

Answer this question based on the above context: {question}
"""


def main():
    # Log in to hugging face 
    login(token=os.environ["HUGGINGFACE_API_KEY"])
    print("Successfully logged in to Hugging Face!")

    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

    


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
    # print(response.text)
    geminiResponse = response.text


    # pipe = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")
    # messages = [
    #     {"role": "user", "content": "Who are you?"},
    # ]
    # pipe(messages)

    # model = Ollama(model="mistral")
    # response_text = model.invoke(prompt)
    response_text = geminiResponse

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()