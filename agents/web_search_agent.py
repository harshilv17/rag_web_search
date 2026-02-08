"""Web Search Agent using Serper API for Google search."""

import os
import requests
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SERPER_API_KEY, SERPER_API_URL, MAX_SEARCH_RESULTS, OPENAI_API_KEY, LLM_MODEL


WEB_SEARCH_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on web search results.

Web Search Results:
{search_results}

User Question: {question}

Instructions:
1. Synthesize the information from the search results to answer the question
2. Cite sources by mentioning the website names when relevant
3. If the search results don't contain enough information, acknowledge the limitations
4. Provide a comprehensive but concise answer

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


def search_serper(query: str, num_results: int = MAX_SEARCH_RESULTS) -> Dict[str, Any]:
    """
    Perform a Google search using Serper API.
    
    Returns:
        Dict containing search results
    """
    if not SERPER_API_KEY:
        raise ValueError("SERPER_API_KEY is not set. Please set it in your .env file.")
    
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": query,
        "num": num_results
    }
    
    try:
        response = requests.post(SERPER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Serper API request failed: {str(e)}")


def format_search_results(results: Dict[str, Any]) -> str:
    """Format Serper API results into a readable string."""
    formatted_parts = []
    
    # Process organic results
    organic = results.get("organic", [])
    for i, result in enumerate(organic, 1):
        title = result.get("title", "No title")
        snippet = result.get("snippet", "No description")
        link = result.get("link", "")
        formatted_parts.append(f"[Result {i}]\nTitle: {title}\nSnippet: {snippet}\nURL: {link}\n")
    
    # Process answer box if present
    answer_box = results.get("answerBox")
    if answer_box:
        answer = answer_box.get("answer") or answer_box.get("snippet", "")
        if answer:
            formatted_parts.insert(0, f"[Featured Answer]\n{answer}\n")
    
    # Process knowledge graph if present
    knowledge_graph = results.get("knowledgeGraph")
    if knowledge_graph:
        title = knowledge_graph.get("title", "")
        description = knowledge_graph.get("description", "")
        if title or description:
            formatted_parts.insert(0, f"[Knowledge Graph]\nTitle: {title}\nDescription: {description}\n")
    
    return "\n".join(formatted_parts) if formatted_parts else "No search results found."


def extract_sources(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract source URLs and titles from search results."""
    sources = []
    
    organic = results.get("organic", [])
    for result in organic:
        sources.append({
            "title": result.get("title", "Unknown"),
            "url": result.get("link", "")
        })
    
    return sources


def search_web(question: str) -> Dict[str, Any]:
    """
    Perform a web search and generate an answer.
    
    Returns:
        Dict containing 'answer', 'sources', and 'raw_results'
    """
    # Perform search
    search_results = search_serper(question)
    
    # Format results for LLM
    formatted_results = format_search_results(search_results)
    
    if formatted_results == "No search results found.":
        return {
            "answer": "I couldn't find any relevant web search results for your query.",
            "sources": [],
            "raw_results": search_results
        }
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template(WEB_SEARCH_PROMPT_TEMPLATE)
    
    # Get LLM and generate response
    llm = get_llm()
    chain = prompt | llm
    
    response = chain.invoke({
        "search_results": formatted_results,
        "question": question
    })
    
    # Extract sources
    sources = extract_sources(search_results)
    
    return {
        "answer": response.content,
        "sources": sources,
        "raw_results": search_results
    }
