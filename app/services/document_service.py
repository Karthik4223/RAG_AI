from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Semantic chunking requirement
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from app.core.config import settings
from app.core.logging import logger
import tempfile
import os

class DocumentService:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=settings.GOOGLE_API_KEY, 
            model=settings.GEMINI_EMBEDDING_MODEL
        )
        
    def process_text(self, title: str, content: str, extra_metadata: Optional[dict] = None) -> List:
        """Processes raw text as a document."""
        metadata = {"source": title}
        if extra_metadata:
            metadata.update(extra_metadata)
            
        doc = Document(page_content=content, metadata=metadata)
        return self.chunk_documents([doc])
        
    def process_file(self, file_path: str, filename: str) -> List:
        """Loads and chunks a document based on its extension."""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
        documents = loader.load()
        logger.info(f"Loaded document {filename} with {len(documents)} pages/sections")
        
        # Add metadata
        for doc in documents:
            doc.metadata["source"] = filename
            
        return self.chunk_documents(documents)

    def chunk_documents(self, documents: List) -> List:
        """
        Splits documents into smaller chunks. 
        Using SemanticChunker for intelligent splitting.
        """
        logger.info("Initializing SemanticChunker...")
        # Note: SemanticChunker uses embeddings to find break points
        text_splitter = SemanticChunker(
            self.embeddings, 
            breakpoint_threshold_type="percentile"
        )
        
        # If semantic chunker is too slow for large docs, fallback to Recursive
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=settings.CHUNK_SIZE,
        #     chunk_overlap=settings.CHUNK_OVERLAP,
        #     add_start_index=True
        # )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} semantic chunks")
        return chunks
