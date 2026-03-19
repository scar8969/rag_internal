# RAG Internal Tool v2

A multi-user RAG (Retrieval-Augmented Generation) system for internal document querying with user authentication, document isolation, and admin management.

## Features

- **Multi-user support**: Up to 20 users with private document spaces
- **User authentication**: JWT-based auth with secure password hashing
- **Document isolation**: Each user sees only their own documents
- **Admin panel**: User management, usage stats, API keys
- **Web interface**: Chat UI for querying documents
- **API access**: REST API with authentication for programmatic access
- **Rate limiting**: Per-user request limits
- **Docker support**: Easy deployment with Docker/Docker Compose

## Quick Start

### 1. Clone and Setup

```bash
cd internal
cp .env.example .env
```

### 2. Configure

Edit `.env` and set:
- `JWT_SECRET` - Generate a long random string
- `OPENAI_API_KEY` - Your OpenAI API key

### 3. Start with Docker

```bash
docker-compose up -d
```

Or without Docker:

```bash
pip install -r requirements.txt
chmod +x start.sh
./start.sh
```

### 4. Create Admin User

On first run, you'll be prompted to create an admin account.

## Usage

### Web Interface

Open http://localhost:8000/app

### CLI Admin Commands

```bash
# Create admin user
python -m scripts.admin create-admin

# Create regular user
python -m scripts.admin create-user

# List users
python -m scripts.admin list-users

# Delete user
python -m scripts.admin delete-user

# Reset password
python -m scripts.admin reset-password
```

### API Endpoints

#### Authentication
- `POST /api/auth/login` - Login
- `POST /api/auth/register` - Register (admin only)
- `GET /api/auth/me` - Get current user

#### RAG Operations
- `POST /api/ingest` - Upload document
- `POST /api/query` - Query with RAG
- `POST /api/search` - Semantic search
- `GET /api/documents` - List documents
- `DELETE /api/documents/{source}` - Delete document

#### User Features
- `GET /api/usage` - Usage stats
- `POST /api/keys` - Create API key
- `GET /api/keys` - List API keys

#### Admin
- `GET /api/admin/users` - List all users
- `DELETE /api/admin/users/{id}` - Delete user
- `GET /api/admin/stats` - System stats

### API Example

```bash
# Login
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"john","password":"secret"}' | jq -r '.access_token')

# Query
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Arduino?"}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Local Server                              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  FastAPI Application                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │   │
│  │  │ Auth     │ │ RAG      │ │ Admin              │  │   │
│  │  │ (JWT)    │ │ Endpoints│ │ Endpoints          │  │   │
│  │  └──────────┘ └──────────┘ └────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                         │                                   │
│         ┌───────────────┼───────────────┐                   │
│         ▼               ▼               ▼                   │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐           │
│  │ ChromaDB  │   │  SQLite   │   │  Redis    │           │
│  │ (Vectors) │   │  (Users)  │   │  (Cache)  │           │
│  └───────────┘   └───────────┘   └───────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./data/app.db` | Database connection |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB directory |
| `JWT_SECRET` | (required) | JWT signing secret |
| `JWT_EXPIRATION_HOURS` | 24 | Token expiration |
| `RATE_LIMIT_PER_MINUTE` | 60 | API rate limit |
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | Embedding model |
| `LLM_MODEL` | `gpt-4` | LLM model |

## Backup

```bash
./backup.sh
```

Backups are stored in `./backups/` with automatic cleanup of old backups.

## Security Notes

1. **Change JWT_SECRET** - Use a long, random string in production
2. **Use HTTPS** - Configure reverse proxy with SSL
3. **Rate limiting** - Adjust per your needs
4. **Passwords** - Use strong passwords

## License

MIT
