import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

from rag.config import Config
from rag.rag import RAGSystem
from .database import Database, User
from .auth import Auth
from .rate_limiter import RateLimiter
from .models import *


config = Config()
database = Database(os.getenv("DATABASE_URL", "sqlite:///./data/app.db").replace("sqlite:///", ""))
auth = Auth(
    database=database,
    jwt_secret=config.jwt_secret,
    jwt_algorithm=config.jwt_algorithm,
    jwt_expiration_hours=config.jwt_expiration_hours,
    jwt_refresh_days=config.jwt_refresh_days,
)
rate_limiter = RateLimiter(requests_per_minute=config.rate_limit_per_minute)

_rag_systems: dict[int, RAGSystem] = {}


def get_rag(user_id: int) -> RAGSystem:
    if user_id not in _rag_systems:
        _rag_systems[user_id] = RAGSystem(config)
    return _rag_systems[user_id]


app = FastAPI(title="RAG Internal Tool API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization", "")
        key = auth_header

        if not key.startswith("Bearer "):
            key = request.client.host if request.client else "unknown"

        result = rate_limiter.check(key)

        if not result.allowed:
            from datetime import datetime
            retry_after = int((result.reset_at - datetime.utcnow()).total_seconds())
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(max(retry_after, 1))}
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        return response


app.add_middleware(RateLimitMiddleware)


def get_current_user(authorization: Optional[str] = Header(None)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.split(" ", 1)[1]
    user = auth.get_user_from_token(token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is disabled")

    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def log_request(user: User, endpoint: str, method: str, status_code: int, latency_ms: int):
    database.log_usage(
        user_id=user.id,
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        latency_ms=latency_ms,
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    return {"status": "ready"}


@app.post("/api/auth/register", response_model=TokenResponse)
def register(request: UserCreate, admin: User = Depends(require_admin)):
    existing = database.get_user_by_username(request.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user = auth.register_user(
        username=request.username,
        password=request.password,
        email=request.email,
    )

    login_result = auth.login(request.username, request.password)
    return TokenResponse(**login_result)


@app.post("/api/auth/login", response_model=TokenResponse)
def login(request: UserLogin):
    result = auth.login(request.username, request.password)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return TokenResponse(**result)


@app.post("/api/auth/refresh", response_model=TokenResponse)
def refresh(request: TokenRefresh):
    result = auth.refresh_access_token(request.refresh_token)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    return TokenResponse(**result)


@app.get("/api/auth/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user)):
    return UserResponse(**user.to_dict())


@app.post("/api/ingest")
async def ingest(
    file: UploadFile,
    user: User = Depends(get_current_user),
):
    start = time.time()

    upload_dir = Path("./data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    temp_path = upload_dir / f"{user.id}_{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        rag = get_rag(user.id)
        result = rag.ingest(str(temp_path), str(user.id))

        temp_path.unlink()

        log_request(user, "/api/ingest", "POST", 200, int((time.time() - start) * 1000))

        return IngestResponse(
            document_id=result.document_id,
            chunks_created=result.chunks_created,
            status=result.status,
        )
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        log_request(user, "/api/ingest", "POST", 500, int((time.time() - start) * 1000))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest, user: User = Depends(get_current_user)):
    start = time.time()

    try:
        rag = get_rag(user.id)
        response = rag.query(
            question=request.question,
            user_id=str(user.id),
            top_k=request.top_k,
            include_sources=request.include_sources,
        )

        log_request(user, "/api/query", "POST", 200, int((time.time() - start) * 1000))

        return QueryResponse(
            answer=response.text,
            sources=response.sources,
            confidence=response.confidence,
        )
    except Exception as e:
        log_request(user, "/api/query", "POST", 500, int((time.time() - start) * 1000))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest, user: User = Depends(get_current_user)):
    start = time.time()

    try:
        rag = get_rag(user.id)
        results = rag.search(
            query=request.query,
            user_id=str(user.id),
            top_k=request.top_k,
        )

        log_request(user, "/api/search", "POST", 200, int((time.time() - start) * 1000))

        return SearchResponse(
            results=[
                SearchResult(
                    text=r.text,
                    source=r.source,
                    distance=r.distance,
                )
                for r in results
            ]
        )
    except Exception as e:
        log_request(user, "/api/search", "POST", 500, int((time.time() - start) * 1000))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
def list_documents(user: User = Depends(get_current_user)):
    rag = get_rag(user.id)
    sources = rag.list_documents(str(user.id))

    return DocumentsResponse(
        documents=[
            DocumentInfo(
                id=src,
                filename=Path(src).name,
                source_path=src,
                file_size=0,
                chunk_count=0,
                created_at="",
            )
            for src in sources
        ],
        count=len(sources),
    )


@app.delete("/api/documents/{source}")
def delete_document(source: str, user: User = Depends(get_current_user)):
    try:
        from urllib.parse import unquote
        source = unquote(source)

        rag = get_rag(user.id)
        rag.delete_document(source, str(user.id))

        return {"status": "deleted", "source": source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/usage", response_model=UsageStats)
def get_usage(user: User = Depends(get_current_user)):
    return UsageStats(**database.get_usage_stats(user.id))


@app.post("/api/keys", response_model=APIKeyResponse)
def create_api_key(request: APIKeyCreate, user: User = Depends(get_current_user)):
    key = database.create_api_key(user.id, request.name)

    keys = database.get_user_api_keys(user.id)
    key_info = next((k for k in keys if k["name"] == request.name), None)

    if not key_info:
        raise HTTPException(status_code=500, detail="Failed to create API key")

    return APIKeyResponse(
        id=key_info["id"],
        name=key_info["name"],
        key=key,
        created_at=key_info["created_at"],
    )


@app.get("/api/keys")
def list_api_keys(user: User = Depends(get_current_user)):
    keys = database.get_user_api_keys(user.id)
    return {"keys": [{"id": k["id"], "name": k["name"], "created_at": k["created_at"]} for k in keys]}


@app.delete("/api/keys/{key_id}")
def revoke_api_key(key_id: str, user: User = Depends(get_current_user)):
    success = database.revoke_api_key(key_id, user.id)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")

    return {"status": "revoked", "key_id": key_id}


@app.get("/api/admin/users", response_model=list[UserResponse])
def list_users(admin: User = Depends(require_admin)):
    users = database.list_users()
    return [UserResponse(**u.to_dict()) for u in users]


@app.delete("/api/admin/users/{user_id}")
def delete_user(user_id: int, admin: User = Depends(require_admin)):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    success = database.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")

    if user_id in _rag_systems:
        del _rag_systems[user_id]

    return {"status": "deleted", "user_id": user_id}


@app.put("/api/admin/users/{user_id}")
def update_user(user_id: int, request: dict, admin: User = Depends(require_admin)):
    success = database.update_user(user_id, **request)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")

    return {"status": "updated", "user_id": user_id}


@app.get("/api/admin/stats")
def admin_stats(admin: User = Depends(require_admin)):
    users = database.list_users()
    total_users = len(users)
    active_users = len([u for u in users if u.is_active])

    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_documents": sum(len(get_rag(u.id).list_documents(str(u.id))) for u in users),
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Internal Tool</title>
        <meta http-equiv="refresh" content="0;url=/app" />
    </head>
    <body>
        <p>Redirecting to <a href="/app">RAG App</a>...</p>
    </body>
    </html>
    """


@app.get("/app", response_class=HTMLResponse)
async def app_page(user: Optional[User] = None):
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Internal Tool</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
            .container { max-width: 900px; margin: 0 auto; padding: 20px; }
            
            #login-screen { display: none; }
            #app-screen { display: none; }
            
            .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            h1 { color: #333; }
            button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            
            input, textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px; }
            textarea { min-height: 100px; resize: vertical; }
            
            .messages { height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 15px; }
            .message { margin-bottom: 15px; padding: 10px 15px; border-radius: 8px; }
            .message.user { background: #007bff; color: white; margin-left: 50px; }
            .message.assistant { background: #e9ecef; margin-right: 50px; }
            .message.system { background: #fff3cd; text-align: center; font-size: 0.9em; }
            .sources { font-size: 0.8em; color: #666; margin-top: 5px; }
            
            .status { display: flex; gap: 20px; font-size: 0.9em; color: #666; }
            .status span { display: flex; align-items: center; gap: 5px; }
            
            .hidden { display: none !important; }
            
            .tabs { display: flex; gap: 10px; margin-bottom: 15px; }
            .tab { padding: 8px 16px; background: #e9ecef; border: none; border-radius: 4px; cursor: pointer; }
            .tab.active { background: #007bff; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <div id="login-screen">
                <div class="card">
                    <h1>RAG Internal Tool</h1>
                    <form id="login-form">
                        <input type="text" id="username" placeholder="Username" required>
                        <input type="password" id="password" placeholder="Password" required>
                        <button type="submit">Login</button>
                    </form>
                    <p id="login-error" style="color: red; margin-top: 10px; display: none;"></p>
                </div>
            </div>
            
            <div id="app-screen">
                <div class="card">
                    <div class="header">
                        <h1>RAG Assistant</h1>
                        <div>
                            <span id="user-info" style="margin-right: 15px;"></span>
                            <button id="logout-btn">Logout</button>
                        </div>
                    </div>
                    
                    <div class="status">
                        <span id="doc-count">Documents: 0</span>
                        <span id="user-stats">Requests today: 0</span>
                    </div>
                </div>
                
                <div class="card">
                    <div class="tabs">
                        <button class="tab active" data-tab="chat">Chat</button>
                        <button class="tab" data-tab="documents">Documents</button>
                    </div>
                    
                    <div id="chat-tab">
                        <div class="messages" id="messages"></div>
                        <form id="query-form">
                            <textarea id="query-input" placeholder="Ask a question about your documents..."></textarea>
                            <button type="submit" id="query-btn">Ask</button>
                        </form>
                    </div>
                    
                    <div id="documents-tab" class="hidden">
                        <div id="documents-list"></div>
                        <h3 style="margin: 20px 0 10px;">Upload Document</h3>
                        <form id="upload-form">
                            <input type="file" id="file-input" accept=".pdf,.txt,.md,.html,.docx">
                            <button type="submit">Upload</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let token = localStorage.getItem('token');
            let currentUser = null;
            
            function showScreen(screen) {
                document.getElementById('login-screen').style.display = screen === 'login' ? 'block' : 'none';
                document.getElementById('app-screen').style.display = screen === 'app' ? 'block' : 'none';
            }
            
            function addMessage(type, content) {
                const div = document.createElement('div');
                div.className = 'message ' + type;
                div.innerHTML = content;
                document.getElementById('messages').appendChild(div);
                document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
            }
            
            async function api(endpoint, options = {}) {
                const headers = { 'Content-Type': 'application/json' };
                if (token) headers['Authorization'] = 'Bearer ' + token;
                
                const response = await fetch(endpoint, {
                    ...options,
                    headers: { ...headers, ...options.headers }
                });
                
                if (response.status === 401) {
                    localStorage.removeItem('token');
                    token = null;
                    showScreen('login');
                    throw new Error('Unauthorized');
                }
                
                return response.json();
            }
            
            async function checkAuth() {
                if (!token) {
                    showScreen('login');
                    return;
                }
                
                try {
                    currentUser = await api('/api/auth/me');
                    document.getElementById('user-info').textContent = currentUser.username + (currentUser.is_admin ? ' (Admin)' : '');
                    showScreen('app');
                    await loadData();
                } catch (e) {
                    showScreen('login');
                }
            }
            
            async function loadData() {
                try {
                    const docs = await api('/api/documents');
                    document.getElementById('doc-count').textContent = 'Documents: ' + docs.count;
                    
                    const usage = await api('/api/usage');
                    document.getElementById('user-stats').textContent = 'Requests (7d): ' + usage.total_requests;
                    
                    const docsList = document.getElementById('documents-list');
                    docsList.innerHTML = docs.documents.length ? 
                        docs.documents.map(d => '<p>' + d.filename + '</p>').join('') :
                        '<p style="color:#666;">No documents uploaded yet.</p>';
                } catch (e) {
                    console.error('Failed to load data:', e);
                }
            }
            
            document.getElementById('login-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                const username = document.getElementById('username').value;
                const password = document.getElementById('password').value;
                
                try {
                    const result = await fetch('/api/auth/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ username, password })
                    }).then(r => r.json());
                    
                    if (result.access_token) {
                        token = result.access_token;
                        localStorage.setItem('token', token);
                        currentUser = result.user;
                        document.getElementById('user-info').textContent = currentUser.username;
                        showScreen('app');
                        await loadData();
                    } else {
                        document.getElementById('login-error').textContent = result.detail || 'Login failed';
                        document.getElementById('login-error').style.display = 'block';
                    }
                } catch (e) {
                    document.getElementById('login-error').textContent = 'Login failed';
                    document.getElementById('login-error').style.display = 'block';
                }
            });
            
            document.getElementById('logout-btn').addEventListener('click', () => {
                token = null;
                localStorage.removeItem('token');
                showScreen('login');
            });
            
            document.getElementById('query-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                const query = document.getElementById('query-input').value.trim();
                if (!query) return;
                
                addMessage('user', query);
                document.getElementById('query-input').value = '';
                
                const btn = document.getElementById('query-btn');
                btn.disabled = true;
                btn.textContent = 'Thinking...';
                
                try {
                    const result = await api('/api/query', {
                        method: 'POST',
                        body: JSON.stringify({ question: query, include_sources: true })
                    });
                    
                    let response = result.answer;
                    if (result.sources && result.sources.length) {
                        response += '<div class="sources">Sources: ' + result.sources.map(s => s.source.split('/').pop()).join(', ') + '</div>';
                    }
                    addMessage('assistant', response);
                } catch (e) {
                    addMessage('assistant', 'Sorry, an error occurred.');
                }
                
                btn.disabled = false;
                btn.textContent = 'Ask';
                await loadData();
            });
            
            document.getElementById('upload-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('file-input');
                if (!fileInput.files.length) return;
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    await api('/api/ingest', {
                        method: 'POST',
                        headers: { 'Authorization': 'Bearer ' + token },
                        body: formData
                    });
                    await loadData();
                    alert('Document uploaded successfully!');
                } catch (e) {
                    alert('Upload failed: ' + e.message);
                }
            });
            
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    
                    const tabName = tab.dataset.tab;
                    document.getElementById('chat-tab').classList.toggle('hidden', tabName !== 'chat');
                    document.getElementById('documents-tab').classList.toggle('hidden', tabName !== 'documents');
                });
            });
            
            checkAuth();
        </script>
    </body>
    </html>
    """
