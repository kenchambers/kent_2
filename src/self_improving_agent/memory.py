from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
import os
from .config import load_api_key

def get_vector_store(path: str) -> FAISS:
    """
    Loads or creates a vector store at the given path.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=load_api_key())
    if Path(path).exists():
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Create a new, empty vector store
        dummy_doc = [Document(page_content="initial document")]
        vector_store = FAISS.from_documents(dummy_doc, embeddings)
        vector_store.save_local(path)
        return vector_store

def add_to_memory(vector_store: FAISS, text: str, path: str):
    """
    Adds a new piece of text to the vector store and saves it.
    """
    vector_store.add_texts([text])
    vector_store.save_local(path)

def query_memory(vector_store: FAISS, query: str) -> str:
    """
    Queries the vector store for relevant information.
    """
    docs = vector_store.similarity_search(query, k=1)
    return docs[0].page_content if docs else "No relevant information found."
