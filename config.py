"""Configuration settings for RAG + Web Search Agent."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Store
CHROMA_PERSIST_DIRECTORY = "data/chroma_db"
COLLECTION_NAME = "documents"

# Serper API
SERPER_API_URL = "https://google.serper.dev/search"
MAX_SEARCH_RESULTS = 5

# RAG Settings
TOP_K_RESULTS = 4
