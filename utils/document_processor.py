"""Document processing utilities for loading and chunking documents."""

import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_text_file(file_path: str) -> str:
    """Load content from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_pdf_file(file_path: str) -> str:
    """Load content from a PDF file."""
    from pypdf import PdfReader
    
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def load_document(file_path: str) -> str:
    """Load document content based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return load_pdf_file(file_path)
    elif ext in ['.txt', '.md']:
        return load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_uploaded_file(uploaded_file) -> str:
    """Load content from a Streamlit uploaded file."""
    file_name = uploaded_file.name
    ext = os.path.splitext(file_name)[1].lower()
    
    if ext == '.pdf':
        from pypdf import PdfReader
        import io
        
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif ext in ['.txt', '.md']:
        return uploaded_file.read().decode('utf-8')
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    metadata: Dict[str, Any] = None
) -> List[Document]:
    """Split text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    documents = []
    for i, chunk in enumerate(chunks):
        doc_metadata = metadata.copy() if metadata else {}
        doc_metadata["chunk_index"] = i
        documents.append(Document(page_content=chunk, metadata=doc_metadata))
    
    return documents


def process_uploaded_file(uploaded_file) -> List[Document]:
    """Process an uploaded file and return chunked documents."""
    text = load_uploaded_file(uploaded_file)
    metadata = {
        "source": uploaded_file.name,
        "file_type": os.path.splitext(uploaded_file.name)[1].lower()
    }
    return chunk_text(text, metadata=metadata)
