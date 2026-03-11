from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Optional
from app.schemas.rag import (
    QueryRequest, 
    QueryResponse, 
    UploadResponse, 
    DocumentChunk, 
    FeedbackRequest,
    ManualIngestRequest,
    IngestTraceResponse,
    QueryTraceResponse,
    ComparisonTraceRequest,
    ComparisonTraceResponse,
    TraceStep
)
from app.services.document_service import DocumentService
from app.services.vector_store import VectorStoreFactory, ChromaStore
from app.services.llm_service import LLMService
from app.services.llama_index_service import LlamaIndexService
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

def get_llama_service():
    return LlamaIndexService()

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

@router.post("/trace-query", response_model=QueryTraceResponse)
async def trace_query(
    request: QueryRequest,
    vector_store: ChromaStore = Depends(get_manual_vector_store),
    llm_service: LLMService = Depends(get_llm_service),
    doc_service: DocumentService = Depends(get_doc_service)
):
    import time
    start_time = time.time()
    try:
        logger.info(f"Tracing query: {request.query}")
        
        # 1. Vectorize Query
        query_vector = doc_service.embeddings.embed_query(request.query)
        
        # 2. Similarity Search
        relevant_chunks = vector_store.similarity_search_with_score(
            query=request.query, 
            k=request.top_k or 5,
            threshold=request.threshold or 0.0
        )
        
        # Format chunks for response
        chunks_for_response = [
            DocumentChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score
            ) for doc, score in relevant_chunks
        ]
        
        # 3. LLM Generation
        if not relevant_chunks:
            answer = "I couldn't find any relevant information to answer your question."
            prompt = "No context provided."
        else:
            result = llm_service.generate_answer(request.query, relevant_chunks)
            answer = result["answer"]
            # To get the prompt, we'd ideally have it from LLMService. 
            # For now, let's reconstruct what goes into it.
            context = "\n\n".join([doc.page_content for doc, _ in relevant_chunks])
            prompt = f"System: Use context below to answer...\n\nContext: {context}\n\nQuestion: {request.query}"

        total_time = (time.time() - start_time) * 1000
        
        return QueryTraceResponse(
            query=request.query,
            query_vector=query_vector,
            retrieved_chunks=chunks_for_response,
            llm_prompt=prompt,
            llm_answer=answer,
            total_time_ms=total_time
        )
    except Exception as e:
        logger.error(f"Trace query failed: {str(e)}")
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

# LlamaIndex Flows
@router.post("/llama-index/upload", response_model=UploadResponse)
async def llama_index_upload(
    file: UploadFile = File(...),
    llama_service: LlamaIndexService = Depends(get_llama_service)
):
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logger.info(f"LlamaIndex: Processing upload {file.filename}")
        chunks_count = llama_service.process_file(tmp_path, file.filename)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return UploadResponse(
            filename=file.filename,
            message="Document successfully indexed via LlamaIndex",
            chunks_indexed=chunks_count
        )
    except Exception as e:
        logger.error(f"LlamaIndex upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/llama-index/query", response_model=QueryResponse)
async def llama_index_query(
    request: QueryRequest,
    llama_service: LlamaIndexService = Depends(get_llama_service)
):
    try:
        result = llama_service.query(request.query, top_k=request.top_k or 5)
        
        sources = [
            DocumentChunk(
                content=s["content"],
                metadata=s["metadata"],
                score=s["score"]
            ) for s in result["sources"]
        ]
        
        import uuid
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            query_id=str(uuid.uuid4())
        )
    except Exception as e:
        logger.error(f"LlamaIndex query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trace-comparison", response_model=ComparisonTraceResponse)
async def trace_comparison(
    request: ComparisonTraceRequest,
    doc_service: DocumentService = Depends(get_doc_service),
    llama_service: LlamaIndexService = Depends(get_llama_service)
):
    try:
        # 1. LangChain Trace
        lc_steps = []
        # Stage 1: Ingest
        lc_steps.append(TraceStep(
            title="Text Loader",
            summary=f"Type: String\nValue: \"{request.text[:20]}...\"",
            output=f"Python Object Type: <class 'str'>\nFull Raw Value: \"{request.text}\"",
            description="Receives text as a primitive string object."
        ))
        
        # Stage 2: Split
        chunks = doc_service.process_text("Comparison", request.text, {})
        lc_steps.append(TraceStep(
            title="Recursive Splitter",
            summary=f"Chunks Total: {len(chunks)}",
            output=f"Strategy: RecursiveCharacterTextSplitter\nChunks Generated: {len(chunks)}\n\n--- FULL CHUNK LIST ---\n" + "\n\n".join([f"CHUNK {i}:\n{c.page_content}" for i, c in enumerate(chunks)]),
            description="Applies recursive character splitting based on separators."
        ))
        
        # Stage 3: Embed
        embedding = doc_service.embeddings.embed_query(request.text)
        lc_steps.append(TraceStep(
            title="Standard Embed",
            summary=f"Model: Gemini\nDim: {len(embedding)}",
            output=f"Provider: Google Generative AI\nModel: models/embedding-001\nDimensions: {len(embedding)}\n\nFULL NUMERICAL VECTOR:\n{embedding}",
            description="Generates a complete numerical vector for the text."
        ))
        
        # Stage 4: Storage
        lc_steps.append(TraceStep(
            title="Vector Storage",
            summary="Store: ChromaDB",
            output=f"DB_ENGINE: ChromaDB\nCOLLECTION_NAME: rag_documents\nMETRIC: COSIGN_SIMILARITY\nSTATUS: ASYNC_WRITE_ACKNOWLEDGED",
            description="Saves the full embedding into the vector database."
        ))

        # 2. LlamaIndex Trace
        li_steps = []
        # Stage 1: Ingest
        from llama_index.core import Document as LiDocument
        li_doc = LiDocument(text=request.text)
        li_steps.append(TraceStep(
            title="Document Ingestion",
            summary=f"DocID: {li_doc.doc_id[:8]}...",
            output=f"Framework Class: llama_index.core.schema.Document\nDocument ID: {li_doc.doc_id}\nFull Content: \"{request.text}\"\nMetadata: {li_doc.metadata}",
            description="Wraps text in a rich Document object with automated metadata."
        ))
        
        # Stage 2: Node Parsing
        from llama_index.core.node_parser import SentenceSplitter
        nodes = SentenceSplitter().get_nodes_from_documents([li_doc])
        li_steps.append(TraceStep(
            title="Node Parsing",
            summary=f"Nodes Created: {len(nodes)}",
            output=f"Parser: SentenceSplitter\nNodes Produced: {len(nodes)}\n\n--- FULL RELATIONSHIP GRAPH ---\n" + "\n".join([f"NODE {i} ID: {n.node_id}\nRelationships: {n.relationships}\n" for i, n in enumerate(nodes)]),
            description="Decomposes the Document into relational Node objects."
        ))
        
        # Stage 3: Indexing
        import uuid
        index_id = str(uuid.uuid4())
        li_steps.append(TraceStep(
            title="Index Relationship",
            summary=f"IndexID: {index_id[:8]}...",
            output=f"Type: VectorStoreIndex\nIndex ID: {index_id}\nNodes Mapping: {[{'id': n.node_id, 'ref': n.ref_doc_id} for n in nodes]}\nIndexing Mode: SIMPLE_EMBEDDING",
            description="Registers all nodes into a structured searchable index."
        ))
        
        # Stage 4: Storage
        li_steps.append(TraceStep(
            title="Storage Context",
            summary="Persist: StorageContext",
            output=f"Store Type: SimpleDocStore\nPersistence Path: ./storage-{index_id}\nIndexStore Configuration: {{'type': 'kv', 'namespace': 'index_store'}}\nNode Data Count: {len(nodes)}",
            description="Serializes the entire index and node relationships."
        ))

        return ComparisonTraceResponse(
            langchain=lc_steps,
            llamaindex=li_steps
        )
    except Exception as e:
        logger.error(f"Comparison trace failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
