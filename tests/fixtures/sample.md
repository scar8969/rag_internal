# RAG System Documentation

## Overview

This document describes the Retrieval-Augmented Generation (RAG) system.

## Features

### Document Processing

The system supports multiple file formats:
- **Text files** (.txt) - Plain text content
- **Markdown** (.md) - Formatted documentation
- **PDF** (.pdf) - Portable Document Format
- **Word** (.docx) - Microsoft Word documents
- **HTML** (.html, .htm) - Web pages

### Embedding Models

The RAG system supports several embedding providers:

| Provider | Model | Dimensions |
|----------|-------|------------|
| OpenAI | text-embedding-ada-002 | 1536 |
| Cohere | embed-multilingual-v3.0 | 1024 |
| HuggingFace | sentence-transformers | 384-768 |
| Ollama | llama2 | 4096 |

### Vector Storage

ChromaDB is used as the vector database for storing document embeddings.

- Persistent storage
- Fast similarity search
- Cosine, Euclidean, and dot product metrics

### Query Processing

1. User submits a question
2. Question is embedded using the same model
3. Semantic search finds relevant chunks
4. Context is assembled and sent to LLM
5. Grounded response with citations is returned

## Configuration

Edit `config.yaml` to customize:
- Embedding provider and model
- Vector store settings
- Retrieval parameters
- Generation settings

## Usage Examples

```bash
# Ingest documents
rag ingest ./docs

# Search
rag search "configuration"

# Query
rag query "How do I set up embedding?"
```