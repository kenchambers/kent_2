"""
This module handles the memory of the agent, including the vector stores.
"""
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from . import config


def get_vector_store(path: str) -> FAISS:
    """
    Loads or creates a vector store at the given path.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=config.load_api_key()
    )
    if Path(path).exists():
        return FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Create a new, empty vector store
    dummy_doc = [Document(page_content="initial document")]
    vector_store = FAISS.from_documents(dummy_doc, embeddings)
    vector_store.save_local(path)
    return vector_store


def get_experience_vector_store() -> FAISS:
    """
    Loads or creates the vector store for the experience layer.
    """
    return get_vector_store(
        str(config.VECTOR_STORES_DIR / "experience_layer.faiss")
    )


def add_to_memory(vector_store: FAISS, text: str, path: str):
    """
    Adds a new piece of text to the vector store and saves it.
    """
    vector_store.add_texts([text])
    vector_store.save_local(path)


def query_memory(vector_store: FAISS, query: str, k: int = 40, threshold: float = None) -> str:
    """
    Queries the vector store for relevant information.
    Returns all highly relevant memories as a formatted string.
    
    Args:
        vector_store: The FAISS vector store to query
        query: The query string
        k: Maximum number of results to retrieve (default 40)
        threshold: Optional similarity threshold (if provided, only returns results above this score)
    """
    # Try to get many relevant memories
    try:
        if threshold:
            # Get results with scores
            docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
            # Filter by threshold (lower scores are better in FAISS)
            docs = [doc for doc, score in docs_and_scores if score <= threshold]
        else:
            docs = vector_store.similarity_search(query, k=k)
    except:
        # Fallback if the vector store is small
        docs = vector_store.similarity_search(query, k=min(k, 5))
    
    if not docs:
        return "No relevant information found."
    
    # Format multiple memories for better context
    if len(docs) == 1:
        return docs[0].page_content
    else:
        memories = []
        for i, doc in enumerate(docs, 1):
            memories.append(f"[Memory {i}]: {doc.page_content}")
        return "\n".join(memories)
