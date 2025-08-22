"""
This module handles the memory of the agent, including the vector stores.
"""
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from typing import List, Dict, Any, Optional
from . import config


def get_vector_store(path: str, initial_documents: Optional[List[Document]] = None) -> FAISS:
    """
    Loads or creates a vector store at the given path.
    If creating a new store, it can be seeded with initial documents.
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

    # If no initial documents are provided, use a default dummy document
    # to ensure the vector store is created correctly.
    documents_to_add = initial_documents if initial_documents else [Document(page_content="__EMPTY_STORE_PLACEHOLDER__")]
    vector_store = FAISS.from_documents(documents_to_add, embeddings)
    vector_store.save_local(path)
    return vector_store


def get_experience_vector_store() -> FAISS:
    """
    Loads or creates the vector store for the experience layer.
    """
    return get_vector_store(
        str(config.VECTOR_STORES_DIR / "experience_layer.faiss")
    )


def add_to_memory(vector_store: FAISS, text: str, path: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Adds a text to the vector store and saves it.
    """
    # FAISS does not support in-place additions well, so we add and re-save.
    vector_store.add_documents([Document(page_content=text, metadata=metadata or {})])
    vector_store.save_local(path)


def query_memory(vector_store: FAISS, query: str, k: int = 40, threshold: float = None) -> List[Document]:
    """
    Queries the vector store for relevant information.
    Returns a list of Document objects, including their metadata.
    
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
    
    # Filter out the default "initial document"
    return [doc for doc in docs if doc.page_content not in ("__EMPTY_STORE_PLACEHOLDER__", "initial document")]
