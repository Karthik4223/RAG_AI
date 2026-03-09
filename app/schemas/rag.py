from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the key findings of the report?")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    metadata_filter: Optional[Dict[str, Any]] = None

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[DocumentChunk]
    query_id: str

class UploadResponse(BaseModel):
    filename: str
    message: str
    chunks_indexed: int

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class ManualIngestRequest(BaseModel):
    title: str = Field(..., example="Project Alpha Overview")
    content: str = Field(..., example="This is the content of Project Alpha...")
    metadata: Optional[Dict[str, Any]] = None
