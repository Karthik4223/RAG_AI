from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from app.core.config import settings
from app.core.logging import logger
from typing import List, Dict, Any, Optional
import os

class LlamaIndexService:
    def __init__(self, collection_name: str = "llama_index_documents"):
        # Configure Gemini
        try:
            Settings.llm = Gemini(
                model_name=settings.GEMINI_MODEL,
                api_key=settings.GOOGLE_API_KEY,
                temperature=0
            )
            Settings.embed_model = GeminiEmbedding(
                model_name=settings.GEMINI_EMBEDDING_MODEL,
                api_key=settings.GOOGLE_API_KEY
            )
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
            self.chroma_collection = self.chroma_client.get_or_create_collection(collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Initialize/Load Index
            # If the collection has data, this will load it
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store, 
                storage_context=self.storage_context
            )
            logger.info(f"Initialized LlamaIndex with collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex: {str(e)}")
            raise

    def process_file(self, file_path: str, filename: str) -> int:
        """Ingests a single file using LlamaIndex."""
        try:
            logger.info(f"LlamaIndex: Processing file {filename}")
            # SimpleDirectoryReader can take a list of files
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            
            # Add metadata to each document
            for doc in documents:
                doc.metadata["source"] = filename
            
            # Insert documents into the index
            for doc in documents:
                self.index.insert(doc)
            
            logger.info(f"LlamaIndex: Ingested {len(documents)} document nodes from {filename}")
            return len(documents)
        except Exception as e:
            logger.error(f"LlamaIndex ingestion failed for {filename}: {str(e)}")
            raise

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Queries the LlamaIndex query engine."""
        try:
            logger.info(f"LlamaIndex: Querying for '{query_text[:50]}...'")
            query_engine = self.index.as_query_engine(similarity_top_k=top_k)
            response = query_engine.query(query_text)
            
            sources = []
            for node in response.source_nodes:
                sources.append({
                    "content": node.node.get_content(),
                    "metadata": node.node.metadata,
                    "score": float(node.score) if node.score is not None else 0.0
                })
            
            return {
                "answer": str(response),
                "sources": sources
            }
        except Exception as e:
            logger.error(f"LlamaIndex query failed: {str(e)}")
            raise
