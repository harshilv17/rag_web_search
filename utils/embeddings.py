"""Embedding utilities using OpenAI embeddings."""

import os
from typing import List
from langchain_openai import OpenAIEmbeddings

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, EMBEDDING_MODEL


def get_embeddings_model() -> OpenAIEmbeddings:
    """Get the OpenAI embeddings model."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please set it in your .env file.")
    
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    embeddings_model = get_embeddings_model()
    return embeddings_model.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query."""
    embeddings_model = get_embeddings_model()
    return embeddings_model.embed_query(query)
