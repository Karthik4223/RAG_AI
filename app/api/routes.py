from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Optional
from app.schemas.rag import (
    QueryRequest, 
    QueryResponse, 
    UploadResponse, 
    DocumentChunk, 
    FeedbackRequest,
    ManualIngestRequest,
    IngestTraceResponse
)
from app.services.document_service import DocumentService
from app.services.vector_store import VectorStoreFactory, ChromaStore
from app.services.llm_service import LLMService
from app.core.logging import logger
import tempfile
import os
import shutil

router = APIRouter()

# Dependency injection
def get_vector_store():
    return VectorStoreFactory.get_vector_store(collection_name="rag_documents")

def get_manual_vector_store():
    return VectorStoreFactory.get_vector_store(collection_name="manual_documents")

def get_doc_service():
    return DocumentService()

def get_llm_service():
    return LLMService()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_service: DocumentService = Depends(get_doc_service),
    vector_store = Depends(get_vector_store)
):
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logger.info(f"Processing uploaded file: {file.filename}")
        chunks = doc_service.process_file(tmp_path, file.filename)
        vector_store.add_documents(chunks)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return UploadResponse(
            filename=file.filename,
            message="Document successfully indexed",
            chunks_indexed=len(chunks)
        )
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest-text", response_model=UploadResponse)
async def ingest_text(
    request: ManualIngestRequest,
    doc_service: DocumentService = Depends(get_doc_service),
    vector_store = Depends(get_manual_vector_store)
):
    try:
        logger.info(f"Processing manual text ingestion: {request.title}")
        chunks = doc_service.process_text(request.title, request.content, request.metadata)
        vector_store.add_documents(chunks)
        
        return UploadResponse(
            filename=request.title,
            message="Text successfully indexed",
            chunks_indexed=len(chunks)
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trace-ingest", response_model=IngestTraceResponse)
async def trace_ingest(
    request: ManualIngestRequest,
    doc_service: DocumentService = Depends(get_doc_service),
    vector_store = Depends(get_manual_vector_store)
):
    try:
        logger.info(f"Tracing ingestion for: {request.title}")
        
        # 1. Chunking
        chunks = doc_service.process_text(request.title, request.content, request.metadata)
        chunk_texts = [c.page_content for c in chunks]
        
        # 2. Embedding (Real call to Google API)
        embeddings = doc_service.embeddings.embed_documents(chunk_texts)
        
        # 3. Storage (Actually add to DB)
        vector_store.add_documents(chunks)
        
        return IngestTraceResponse(
            title=request.title,
            raw_content=request.content,
            chunks=chunk_texts,
            vectors=embeddings,
            metadata=request.metadata or {},
            total_chunks=len(chunks)
        )
    except Exception as e:
        logger.error(f"Trace ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    vector_store: ChromaStore = Depends(get_vector_store),
    llm_service: LLMService = Depends(get_llm_service)
):
    try:
        # 1. Similarity Search with threshold
        relevant_chunks = vector_store.similarity_search_with_score(
            query=request.query, 
            k=request.top_k or 5,
            threshold=request.threshold or 0.0
        )
        
        if not relevant_chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information in the documents to answer your question.",
                sources=[],
                query_id="none"
            )

        # 2. Generate Answer
        result = llm_service.generate_answer(request.query, relevant_chunks)
        
        # 3. Format sources
        sources = [
            DocumentChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score
            ) for doc, score in relevant_chunks
        ]
        
        # Log interaction (could be saved to DB here for feedback loop)
        logger.info(f"Query: {request.query} | Response ID: {result['query_id']}")
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            query_id=result["query_id"]
        )
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    # In a production system, this would store feedback in a database (e.g., PostgreSQL)
    # to be used later for fine-tuning or prompt optimization.
    logger.info(f"Received feedback for {request.query_id}: Rating {request.rating}, Comment: {request.comment}")
    return {"status": "success", "message": "Feedback received"}
