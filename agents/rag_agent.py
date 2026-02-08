"""RAG Agent for document-based question answering."""

import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, LLM_MODEL, TOP_K_RESULTS
from utils.vector_store import similarity_search, get_document_count


RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on the provided context from documents.

Context from documents:
{context}

User Question: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information to answer, say so clearly
3. Cite which document/source the information comes from when relevant
4. Be concise but thorough

Answer:"""


def get_llm() -> ChatOpenAI:
    """Get the ChatOpenAI LLM instance."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please set it in your .env file.")
    
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7
    )


def format_context(documents: List[Document]) -> str:
    """Format retrieved documents into a context string."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[Document {i} - Source: {source}]\n{doc.page_content}\n")
    return "\n".join(context_parts)


def query_rag(question: str, k: int = TOP_K_RESULTS) -> Dict[str, Any]:
    """
    Query the RAG system with a question.
    
    Returns:
        Dict containing 'answer', 'sources', and 'context'
    """
    # Check if we have any documents
    doc_count = get_document_count()
    if doc_count == 0:
        return {
            "answer": "No documents have been uploaded yet. Please upload some documents first to use RAG mode.",
            "sources": [],
            "context": ""
        }
    
    # Retrieve relevant documents
    retrieved_docs = similarity_search(question, k=k)
    
    if not retrieved_docs:
        return {
            "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
            "sources": [],
            "context": ""
        }
    
    # Format context
    context = format_context(retrieved_docs)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # Get LLM and generate response
    llm = get_llm()
    chain = prompt | llm
    
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    # Extract sources
    sources = list(set(
        doc.metadata.get("source", "Unknown") 
        for doc in retrieved_docs
    ))
    
    return {
        "answer": response.content,
        "sources": sources,
        "context": context
    }
