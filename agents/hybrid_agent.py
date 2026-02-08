"""Hybrid Agent that combines RAG and Web Search capabilities."""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, LLM_MODEL
from agents.rag_agent import query_rag
from agents.web_search_agent import search_web
from utils.vector_store import get_document_count


HYBRID_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions using both document knowledge and web search results.

Document Context (from uploaded documents):
{rag_context}

Web Search Results:
{web_context}

User Question: {question}

Instructions:
1. Synthesize information from BOTH the documents and web search results
2. Prioritize document information for specific/local knowledge
3. Use web search results for broader context or recent information
4. Clearly indicate when information comes from documents vs. the web
5. If there are conflicts between sources, acknowledge them
6. Be comprehensive but concise

Answer:"""


QUERY_ROUTER_PROMPT = """Analyze this question and determine the best way to answer it.

Question: {question}

Context:
- Documents available: {has_documents}
- Types of documents: {document_sources}

Determine if this question is best answered by:
1. "documents" - if it's about specific content in the uploaded documents
2. "web" - if it requires recent/external information not likely in documents
3. "hybrid" - if it could benefit from both document context and web information

Respond with just one word: documents, web, or hybrid"""


def get_llm() -> ChatOpenAI:
    """Get the ChatOpenAI LLM instance."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please set it in your .env file.")
    
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7
    )


def route_query(question: str) -> str:
    """
    Route a query to the appropriate agent(s).
    
    Returns:
        'documents', 'web', or 'hybrid'
    """
    from utils.vector_store import get_all_sources
    
    doc_count = get_document_count()
    has_documents = doc_count > 0
    sources = get_all_sources() if has_documents else []
    
    # If no documents, use web search
    if not has_documents:
        return "web"
    
    # Use LLM to route
    prompt = ChatPromptTemplate.from_template(QUERY_ROUTER_PROMPT)
    llm = get_llm()
    chain = prompt | llm
    
    response = chain.invoke({
        "question": question,
        "has_documents": "Yes" if has_documents else "No",
        "document_sources": ", ".join(sources) if sources else "None"
    })
    
    route = response.content.strip().lower()
    
    # Validate response
    if route in ["documents", "web", "hybrid"]:
        return route
    else:
        # Default to hybrid if unclear
        return "hybrid"


def query_hybrid(question: str, auto_route: bool = True) -> Dict[str, Any]:
    """
    Query using both RAG and Web Search, combining the results.
    
    Args:
        question: The user's question
        auto_route: If True, automatically determine the best approach
    
    Returns:
        Dict containing 'answer', 'sources', 'mode', and context info
    """
    doc_count = get_document_count()
    
    if auto_route:
        route = route_query(question)
        
        if route == "documents":
            result = query_rag(question)
            result["mode"] = "documents"
            return result
        elif route == "web":
            result = search_web(question)
            result["mode"] = "web"
            return result
    
    # Hybrid mode: combine both
    rag_result = {"answer": "", "sources": [], "context": ""}
    web_result = {"answer": "", "sources": [], "raw_results": {}}
    
    # Get RAG results if documents available
    if doc_count > 0:
        try:
            rag_result = query_rag(question)
        except Exception as e:
            rag_result["context"] = f"Error retrieving from documents: {str(e)}"
    
    # Get web search results
    try:
        web_result = search_web(question)
    except Exception as e:
        web_result["answer"] = f"Error searching web: {str(e)}"
    
    # Combine using LLM
    rag_context = rag_result.get("context", "No document context available.")
    web_context = web_result.get("answer", "No web search results available.")
    
    prompt = ChatPromptTemplate.from_template(HYBRID_PROMPT_TEMPLATE)
    llm = get_llm()
    chain = prompt | llm
    
    response = chain.invoke({
        "rag_context": rag_context,
        "web_context": web_context,
        "question": question
    })
    
    # Combine sources
    all_sources = []
    
    # Add document sources
    for source in rag_result.get("sources", []):
        all_sources.append({"type": "document", "name": source})
    
    # Add web sources
    for source in web_result.get("sources", []):
        all_sources.append({
            "type": "web",
            "title": source.get("title", ""),
            "url": source.get("url", "")
        })
    
    return {
        "answer": response.content,
        "sources": all_sources,
        "mode": "hybrid",
        "rag_context": rag_context,
        "web_results": web_result.get("raw_results", {})
    }
