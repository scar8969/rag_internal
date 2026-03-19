import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "internal"))

from rag.rag import RAGSystem


def main():
    parser = argparse.ArgumentParser(description="RAG CLI - Document Question Answering")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="File or directory path to ingest")
    ingest_parser.add_argument("--user-id", default="default", help="User ID for isolation")

    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--user-id", default="default", help="User ID for isolation")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--user-id", default="default", help="User ID for isolation")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    list_parser = subparsers.add_parser("list", help="List ingested documents")
    list_parser.add_argument("--user-id", default="default", help="User ID for isolation")

    delete_parser = subparsers.add_parser("delete", help="Delete a document")
    delete_parser.add_argument("source", help="Source path to delete")
    delete_parser.add_argument("--user-id", default="default", help="User ID for isolation")

    args = parser.parse_args()

    rag = RAGSystem()

    if args.command == "ingest":
        path = Path(args.path)
        if path.is_dir():
            print(f"Ingesting directory: {args.path}")
            results = rag.ingest_directory(args.path, args.user_id)
            for result in results:
                print(f"  - {result.document_id}: {result.chunks_created} chunks")
        else:
            print(f"Ingesting file: {args.path}")
            result = rag.ingest(args.path, args.user_id)
            print(f"Document ID: {result.document_id}")
            print(f"Chunks created: {result.chunks_created}")
            print(f"Status: {result.status}")

    elif args.command == "query":
        print(f"Question: {args.question}")
        print("-" * 50)
        response = rag.query(args.question, args.user_id, args.top_k)
        print(response.text)
        if response.sources:
            print("\nSources:")
            for source in response.sources:
                print(f"  - {source}")

    elif args.command == "search":
        print(f"Query: {args.query}")
        print("-" * 50)
        results = rag.search(args.query, args.user_id, args.top_k)
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Score: {1 - result.distance:.4f}")
            print(f"    {result.text[:200]}...")

    elif args.command == "list":
        sources = rag.list_documents(args.user_id)
        if sources:
            print(f"Documents ({len(sources)}):")
            for src in sources:
                print(f"  - {src}")
        else:
            print("No documents ingested yet.")

    elif args.command == "delete":
        rag.delete_document(args.source, args.user_id)
        print(f"Deleted: {args.source}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
