
import streamlit as st
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Lazy imports to avoid chromadb pydantic issues at startup
# These functions will be imported when actually needed

def get_document_processor():
    from utils.document_processor import process_uploaded_file
    return process_uploaded_file

def get_vector_store_functions():
    from utils.vector_store import (
        add_documents, 
        get_document_count, 
        get_all_sources, 
        delete_documents_by_source,
        clear_all_documents
    )
    return add_documents, get_document_count, get_all_sources, delete_documents_by_source, clear_all_documents

def get_agents():
    from agents.rag_agent import query_rag
    from agents.web_search_agent import search_web
    from agents.hybrid_agent import query_hybrid
    return query_rag, search_web, query_hybrid

# Page configuration
st.set_page_config(
    page_title="RAG + Web Search Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    /* Modern dark theme adjustments */
    .stApp {
        background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%);
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 1rem 1rem 0 1rem;
        margin: 0.5rem 0;
        color: white;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1e2330 0%, #2d3548 100%);
        padding: 1rem 1.5rem;
        border-radius: 1rem 1rem 1rem 0;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Source badges */
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.5);
        border-radius: 1rem;
        font-size: 0.8rem;
        margin: 0.25rem;
    }
    
    /* Mode selector styling */
    .mode-card {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 0.5rem;
    }
    
    /* Document list styling */
    .doc-item {
        background: rgba(255,255,255,0.05);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(0,0,0,0.2);
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
    }
    
    .status-inactive {
        background: #6b7280;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = "hybrid"


def render_sidebar():
    """Render the sidebar with configuration and document management."""
    with st.sidebar:
        st.markdown("## 🤖 Agent Configuration")
        
        # Mode selection
        st.markdown("### Query Mode")
        mode_options = {
            "hybrid": "🔀 Hybrid (Auto-route)",
            "rag": "📚 RAG Only",
            "web": "🌐 Web Search Only"
        }
        
        selected_mode = st.radio(
            "Select how queries are processed:",
            options=list(mode_options.keys()),
            format_func=lambda x: mode_options[x],
            index=list(mode_options.keys()).index(st.session_state.mode),
            key="mode_selector"
        )
        st.session_state.mode = selected_mode
        
        # Mode descriptions
        mode_descriptions = {
            "hybrid": "Intelligently combines document knowledge and web search",
            "rag": "Answers only from uploaded documents",
            "web": "Searches the web for answers using Serper API"
        }
        st.caption(mode_descriptions[selected_mode])
        
        st.markdown("---")
        
        # Document upload section
        st.markdown("### 📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents for RAG",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.button("📤 Process Documents", use_container_width=True):
                with st.spinner("Processing documents..."):
                    # Lazy import
                    process_uploaded_file = get_document_processor()
                    add_documents, _, _, _, _ = get_vector_store_functions()
                    for uploaded_file in uploaded_files:
                        try:
                            documents = process_uploaded_file(uploaded_file)
                            add_documents(documents)
                            st.success(f"✅ Processed: {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        st.markdown("---")
        
        # Document management
        st.markdown("### 📁 Uploaded Documents")
        
        # Lazy imports for vector store functions
        try:
            _, get_document_count, get_all_sources, delete_documents_by_source, clear_all_documents = get_vector_store_functions()
            doc_count = get_document_count()
            sources = get_all_sources()
        except Exception:
            doc_count = 0
            sources = []
        
        if doc_count > 0:
            st.markdown(f"**{doc_count}** chunks from **{len(sources)}** document(s)")
            
            for source in sources:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"📄 {source}")
                with col2:
                    if st.button("🗑️", key=f"delete_{source}"):
                        delete_documents_by_source(source)
                        st.rerun()
            
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                clear_all_documents()
                st.rerun()
        else:
            st.info("No documents uploaded yet")
        
        st.markdown("---")
        
        # API Status
        st.markdown("### 🔑 API Status")
        
        from config import SERPER_API_KEY, OPENAI_API_KEY
        
        serper_status = "✅ Configured" if SERPER_API_KEY else "❌ Not set"
        openai_status = "✅ Configured" if OPENAI_API_KEY else "❌ Not set"
        
        st.markdown(f"**Serper API:** {serper_status}")
        st.markdown(f"**OpenAI API:** {openai_status}")
        
        if not SERPER_API_KEY or not OPENAI_API_KEY:
            st.warning("Please set API keys in your .env file")


def render_chat_interface():
    """Render the main chat interface."""
    st.markdown("# 🤖 RAG + Web Search Agent")
    st.markdown("*Combining document knowledge with real-time web search*")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources", expanded=False):
                        for source in message["sources"]:
                            if isinstance(source, dict):
                                if source.get("type") == "web":
                                    st.markdown(f"🌐 [{source.get('title', 'Web')}]({source.get('url', '')})")
                                else:
                                    st.markdown(f"📄 {source.get('name', source)}")
                            else:
                                st.markdown(f"📄 {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    mode = st.session_state.mode
                    
                    # Lazy import agents
                    query_rag, search_web, query_hybrid = get_agents()
                    
                    if mode == "rag":
                        result = query_rag(prompt)
                    elif mode == "web":
                        result = search_web(prompt)
                    else:  # hybrid
                        result = query_hybrid(prompt, auto_route=True)
                    
                    answer = result.get("answer", "I couldn't generate a response.")
                    sources = result.get("sources", [])
                    used_mode = result.get("mode", mode)
                    
                    # Display mode indicator
                    mode_icons = {"documents": "📚", "web": "🌐", "hybrid": "🔀"}
                    mode_icon = mode_icons.get(used_mode, "🤖")
                    
                    st.markdown(f"*{mode_icon} Mode: {used_mode.title()}*")
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for source in sources:
                                if isinstance(source, dict):
                                    if source.get("type") == "web":
                                        st.markdown(f"🌐 [{source.get('title', 'Web')}]({source.get('url', '')})")
                                    else:
                                        st.markdown(f"📄 {source.get('name', source)}")
                                else:
                                    st.markdown(f"📄 {source}")
                    
                    # Save to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "mode": used_mode
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })


def main():
    """Main application entry point."""
    initialize_session_state()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
