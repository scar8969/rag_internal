# RAG Internal

A production-ready **Retrieval-Augmented Generation (RAG)** system for building intelligent document question-answering applications.

## Features

- **Semantic Search** - Find relevant content by meaning, not just keywords
- **Multi-format Support** - PDF, Word, Markdown, Text, HTML files
- **Source Citations** - Every answer includes references to source documents
- **Advanced Retrieval** - Reranking, hybrid search, query expansion
- **Multiple LLM Providers** - OpenAI GPT-4, Anthropic Claude, Ollama (local)
- **Multi-user Support** - User authentication with JWT tokens
- **Document Isolation** - Each user has private document space
- **Docker Ready** - Deploy anywhere with Docker

## Project Structure

```
rag/
├── internal/              # Full API server with auth
│   ├── api/               # FastAPI endpoints
│   │   ├── auth.py        # JWT authentication
│   │   ├── database.py    # User management
│   │   ├── main.py        # API routes
│   │   ├── models.py      # Pydantic models
│   │   └── rate_limiter.py
│   ├── rag/               # RAG core modules
│   │   ├── chunker.py     # Document splitting
│   │   ├── config.py      # Configuration
│   │   ├── document_loader.py
│   │   ├── embedder.py    # Embedding generation
│   │   ├── generator.py   # LLM integration
│   │   ├── rag.py         # Main RAG system
│   │   ├── retriever.py   # Search/retrieval
│   │   └── vector_store.py # ChromaDB integration
│   ├── scripts/           # Admin utilities
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── tests/                 # Test suite
├── main.py               # Standalone CLI (no auth)
├── Dockerfile           # CLI Docker image
├── docker-compose.yml    # CLI Docker compose
└── .env.example
```

## Quick Start

### Option 1: Internal API Server (with auth)

```bash
cd internal

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Start with Docker
docker compose up -d

# Or without Docker
pip install -r requirements.txt
python run.py
```

Access at http://localhost:8001

### Option 2: Standalone CLI

```bash
# Install dependencies
pip install -r internal/requirements.txt

# Ingest documents
python main.py ingest ./documents

# Query
python main.py query "What is this about?"

# Search
python main.py search "Arduino"
```

### Option 3: Docker CLI

```bash
docker compose up -d
docker compose run --rm rag-cli ingest ./documents
docker compose run --rm rag-cli query "Your question"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional |
| `OLLAMA_BASE_URL` | Ollama server URL | http://localhost:11434 |
| `EMBEDDING_PROVIDER` | openai, anthropic, ollama | openai |
| `LLM_PROVIDER` | openai, anthropic, ollama | openai |
| `JWT_SECRET` | Secret for JWT tokens | change-me |
| `CHROMA_PERSIST_DIR` | Vector database directory | ./data/chroma |

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register user (admin only)
- `POST /api/auth/login` - Login
- `POST /api/auth/refresh` - Refresh token
- `GET /api/auth/me` - Get current user

### RAG Operations
- `POST /api/ingest` - Upload and index document
- `POST /api/query` - Ask question
- `POST /api/search` - Semantic search
- `GET /api/documents` - List documents
- `DELETE /api/documents/{source}` - Delete document

### Admin (requires admin user)
- `GET /api/admin/users` - List users
- `DELETE /api/admin/users/{id}` - Delete user
- `GET /api/admin/stats` - System statistics

## CLI Commands

```bash
# Ingest
python main.py ingest <file_or_directory>
python main.py ingest ./documents --user-id myuser

# Query
python main.py query "What is this about?"
python main.py query "Explain Arduino" --top-k 10

# Search
python main.py search <query>
python main.py search "keywords" --top-k 5

# List documents
python main.py list --user-id myuser

# Delete document
python main.py delete <source_path> --user-id myuser
```

## Supported File Types

- PDF (.pdf)
- Word (.docx)
- Markdown (.md)
- Plain text (.txt)
- HTML (.html)

## Architecture

```
Document → Loader → Chunker → Embedder → Vector Store (ChromaDB)
                                                       ↓
Query → Embedder → Semantic Search → LLM Generation → Answer
```

## Testing

```bash
cd internal
pytest tests/ -v
```

## Backup

```bash
cd internal
./backup.sh
```

Backups stored in `internal/backups/` with automatic cleanup.

## Docker Commands

```bash
# Build images
docker compose build

# Start services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Shell into container
docker compose exec rag-api bash

# Restart
docker compose restart
```

## License

MIT
