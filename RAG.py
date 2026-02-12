import os
from supabase import create_client
from langchain.embeddings import ReplicateEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from pypdf import PdfReader




supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


embeddings = ReplicateEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    replicate_api_token=os.environ["REPLICATE_API_TOKEN"]
)

vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents"
)

llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3-8b-instruct",
    model_kwargs={"temperature": 0.1},
    huggingfacehub_api_token=os.environ["HUGGINGFACE_API_KEY"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

reader = PdfReader("document.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

vectorstore.add_texts([text])


# Query
print (qa.run("What are key points about data handling?"))


