#!/bin/bash

set -e

echo "========================================"
echo "RAG Internal Tool - Startup"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "IMPORTANT: Edit .env and set JWT_SECRET and OPENAI_API_KEY"
    echo ""
fi

if [ ! -f data/app.db ]; then
    echo "Initializing database..."
    python3 -m scripts.admin init-db
fi

echo "Checking for admin user..."
ADMIN_EXISTS=$(python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.admin import Database, Config
import os
db = Database(os.getenv('DATABASE_URL', 'sqlite:///./data/app.db').replace('sqlite:///', ''))
users = db.list_users()
print('yes' if any(u.is_admin for u in users) else 'no')
" 2>/dev/null || echo "no")

if [ "$ADMIN_EXISTS" != "yes" ]; then
    echo ""
    echo "No admin user found. Creating admin account..."
    python3 -m scripts.admin create-admin
fi

echo ""
echo "Starting RAG Internal Tool..."
echo "Open http://localhost:8000 in your browser"
echo ""

exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
