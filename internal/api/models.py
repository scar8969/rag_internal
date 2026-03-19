from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    is_admin: bool = False


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str]
    created_at: str
    is_admin: bool
    is_active: bool
    daily_limit: int


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: Optional[UserResponse] = None


class TokenRefresh(BaseModel):
    refresh_token: str


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    include_sources: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class SearchResult(BaseModel):
    text: str
    source: str
    distance: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class IngestRequest(BaseModel):
    source: str
    metadata: Optional[dict] = None


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    status: str


class DocumentInfo(BaseModel):
    id: str
    filename: str
    source_path: str
    file_size: int
    chunk_count: int
    created_at: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    count: int


class UsageStats(BaseModel):
    total_requests: int
    total_tokens: int
    active_days: int


class APIKeyCreate(BaseModel):
    name: str


class APIKeyResponse(BaseModel):
    id: str
    name: str
    key: str
    created_at: str


class StatusResponse(BaseModel):
    status: str
    documents: int
    users: int
