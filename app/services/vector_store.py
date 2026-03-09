from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma as LangChainChroma
from app.core.config import settings
from app.core.logging import logger
import os

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Any]):
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Any]:
        pass

class VectorStoreFactory:
    @staticmethod
    def get_vector_store(collection_name: str = "rag_documents") -> BaseVectorStore:
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=settings.GOOGLE_API_KEY, 
            model=settings.GEMINI_EMBEDDING_MODEL
        )
        
        if settings.VECTOR_STORE_TYPE == "chroma":
            return ChromaStore(embeddings, collection_name=collection_name)
        elif settings.VECTOR_STORE_TYPE == "pinecone":
            # Implementation for Pinecone would go here
            # return PineconeStore(embeddings)
            raise NotImplementedError("Pinecone integration not implemented yet in this demo")
        else:
            raise ValueError(f"Unsupported vector store type: {settings.VECTOR_STORE_TYPE}")

class ChromaStore(BaseVectorStore):
    def __init__(self, embeddings, collection_name: str = "rag_documents"):
        self.embeddings = embeddings
        self.db = LangChainChroma(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        logger.info(f"Initialized ChromaDB at {settings.CHROMA_PERSIST_DIRECTORY} with collection {collection_name}")

    def add_documents(self, documents: List[Any]):
        try:
            self.db.add_documents(documents)
            logger.info(f"Added {len(documents)} document chunks to ChromaDB")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise

    def similarity_search_with_score(self, query: str, k: int, threshold: float = 0.0) -> List[tuple]:
        try:
            results = self.db.similarity_search_with_relevance_scores(query, k=k)
            
            # Log all retrieved scores for debugging
            for i, (doc, score) in enumerate(results):
                logger.debug(f"Rank {i+1}: Score {score:.4f} | Content: {doc.page_content[:50]}...")

            # Filter by threshold
            filtered_results = [(doc, score) for doc, score in results if score >= threshold]
            logger.info(f"Filtered {len(filtered_results)} results above threshold {threshold} (Top score: {results[0][1] if results else 'N/A'})")
            return filtered_results
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int, filter: Optional[Dict[str, Any]] = None) -> List[Any]:
        # Standard implementation
        return self.db.similarity_search(query, k=k, filter=filter)
