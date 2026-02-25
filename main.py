#this code is just a template that is used to create a simple API using FastAPI. 
# It defines two endpoints: the root endpoint ("/") which returns a simple
# greeting message, and another endpoint ("/items/{item_id}") which takes
# an item ID as a path parameter and an optional query parameter "q".
# The response includes the item ID and the value of "q" if it is provided.

from fastapi import FastAPI, HTTPException, Query
from query_database import get_top_k_matches

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}git


@app.get("/search")
def search(q: str = Query(..., title="Query string"), k: int = Query(1, ge=1, le=20)):
    """Search the Supabase vector store for the most similar documents.

    This reuses `get_top_k_matches` from `query_database.py` which will initialize
    the embeddings and vector store as needed.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        results = get_top_k_matches(q, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"query": q, "k": k, "results": results}