import os
from dotenv import load_dotenv
import google.generativeai as genai
from doc_fetcher import fetch_similar_docs, load_embeddings
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

model_name = "sentence-transformers/all-MiniLM-L6-v2" 
HF_MODEL = None
# # Set up Hugging Face pipeline for conversation (change model_name for chat models)
# def get_hf_conversation_pipeline():
#     return pipeline("text-generation", model=model_name, token=hf_access_token)

# hf_conversation = get_hf_conversation_pipeline()

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
hf_access_token = os.getenv("HF_KEY")

genai.configure(api_key=gemini_key)
model = genai.GenerativeModel("gemini-1.5-flash")

load_embeddings()


def get_query_embedding(query):
    """Return embedding for a query string using Hugging Face model."""
    global HF_MODEL
    if HF_MODEL is None:
        HF_MODEL = SentenceTransformer(model_name, use_auth_token=hf_access_token)
    emb = HF_MODEL.encode([query], convert_to_numpy=True)[0]
    return emb.astype(np.float32)


# def get_embedding(text: str):
#     result = genai.embed_content(
#         model="models/embedding-001",  # Gemini embedding model
#         content=text
#     )
#     return np.array(result["embedding"], dtype="float32")

def ask_gemini(query):
    query_embedding = get_query_embedding(query)
    docs = fetch_similar_docs(query_embedding, top_k=5)
    context = "\n".join(docs) if docs else "No relevant documents found."

    prompt = f"""
    You are a helpful assistant.
    The user asked: {query}
    Here are some database documents:
    {context}
    
    Use these documents to answer the user in a conversational way.
    """

    response = model.generate_content(prompt)
    return response.text

# # Hugging Face conversational LLM function
# def ask_huggingface(query):
#     docs = fetch_similar_docs(get_embedding(query), top_k=5)
#     context = "\n".join(docs) if docs else "No relevant documents found."
#     prompt = f"You are a helpful assistant. The user asked: {query}\nHere are some database documents:\n{context}\nUse these documents to answer the user in a conversational way."
#     result = hf_conversation(prompt, max_length=512, do_sample=True)
#     return result[0]['generated_text']
