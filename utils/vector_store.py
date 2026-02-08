"""Vector store utilities using ChromaDB with Python 3.14 compatibility."""

import os
from typing import List
from langchain_core.documents import Document

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME, TOP_K_RESULTS


# Global variable to cache the vector store
_vector_store = None


def get_embeddings_model():
    """Lazy import of embeddings to avoid early chromadb import."""
    from utils.embeddings import get_embeddings_model as _get_embeddings
    return _get_embeddings()


def get_vector_store():
    """Get or create the ChromaDB vector store with lazy loading."""
    global _vector_store
    
    if _vector_store is not None:
        return _vector_store
    
    # Ensure the persist directory exists
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    
    # Lazy import to avoid early initialization issues
    from langchain_chroma import Chroma
    
    embeddings = get_embeddings_model()
    
    _vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    
    return _vector_store


def add_documents(documents: List[Document]) -> None:
    """Add documents to the vector store."""
    vector_store = get_vector_store()
    vector_store.add_documents(documents)


def similarity_search(query: str, k: int = TOP_K_RESULTS) -> List[Document]:
    """Perform similarity search on the vector store."""
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k)


def get_document_count() -> int:
    """Get the number of documents in the vector store."""
    try:
        vector_store = get_vector_store()
        return vector_store._collection.count()
    except Exception:
        return 0


def get_all_sources() -> List[str]:
    """Get all unique source names from the vector store."""
    try:
        vector_store = get_vector_store()
        
        # Get all documents metadata
        collection = vector_store._collection
        results = collection.get(include=["metadatas"])
        
        sources = set()
        if results and results.get("metadatas"):
            for metadata in results["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])
        
        return list(sources)
    except Exception:
        return []


def delete_documents_by_source(source: str) -> None:
    """Delete all documents with a specific source."""
    vector_store = get_vector_store()
    collection = vector_store._collection
    
    # Get IDs of documents with this source
    results = collection.get(include=["metadatas"])
    
    ids_to_delete = []
    if results and results.get("ids") and results.get("metadatas"):
        for doc_id, metadata in zip(results["ids"], results["metadatas"]):
            if metadata and metadata.get("source") == source:
                ids_to_delete.append(doc_id)
    
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)


def clear_all_documents() -> None:
    """Clear all documents from the vector store."""
    vector_store = get_vector_store()
    collection = vector_store._collection
    
    # Get all IDs and delete them
    results = collection.get()
    if results and results.get("ids"):
        collection.delete(ids=results["ids"])
