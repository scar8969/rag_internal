#!/bin/bash
set -e

cd "$(dirname "$0")/internal"

echo "Building Docker images..."
docker compose build

echo ""
echo "Starting RAG API..."
docker compose up -d

echo ""
echo "RAG API is running at http://localhost:8000"
echo ""
echo "Useful commands:"
echo "  docker compose logs -f    # View logs"
echo "  docker compose down      # Stop services"
echo "  docker compose exec rag-api python run.py  # Run shell in container"
