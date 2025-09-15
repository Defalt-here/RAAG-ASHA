
import os
import psycopg2
import numpy as np
import faiss
from dotenv import load_dotenv
from TouchRDS import get_connection
load_dotenv()

# Global FAISS index
INDEX = None
DOCS = []

def load_embeddings():
    """Load all embeddings + docs from Postgres into FAISS."""
    global INDEX, DOCS

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT text, embedding FROM documents;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("⚠️ No rows found in DB!")
        return

    # Extract embeddings + docs

    DOCS = [(r[0], r[1]) for r in rows]  # (id, content)
    # Convert embedding bytes to numpy float32 array
    embeddings = [np.frombuffer(r[1], dtype="float32") for r in rows]
    embeddings = np.stack(embeddings)

    dim = embeddings.shape[1]  # 1536 for OpenAI/Gemini
    INDEX = faiss.IndexFlatL2(dim)
    INDEX.add(embeddings)
    print(f"✅ Loaded {len(rows)} embeddings into FAISS")

def fetch_similar_docs(query_embedding, top_k=5):
    """Return top_k docs most similar to the query embedding."""
    global INDEX, DOCS
    if INDEX is None:
        raise RuntimeError("FAISS index is empty. Call load_embeddings() first.")

    query_embedding = np.array([query_embedding]).astype("float32")
    distances, indices = INDEX.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(DOCS):
            results.append(DOCS[idx][0])  # return text only
    return results
