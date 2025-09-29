#!/usr/bin/env python3
"""
Training CLI for Graph RAG system.
Creates and saves knowledge graphs from documents.
"""

import argparse
import sys
from pathlib import Path

from src.graph_rag.core import GraphRAGTrainer
from src.graph_rag.config import settings


def main():
    parser = argparse.ArgumentParser(description="Train Graph RAG system from documents")
    parser.add_argument("document_path", help="Path to the document file to process")
    parser.add_argument("--graph-name", "-n",
                       help=f"Name for the knowledge graph (default: '{settings.DEFAULT_GRAPH_NAME}')")
    parser.add_argument("--storage-dir", "-s",
                       help=f"Directory to store the knowledge graph (default: '{settings.DEFAULT_STORAGE_DIR}')")
    parser.add_argument("--neo4j", action="store_true", help="Use Neo4j graph store")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable detailed logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Disable verbose logging")

    args = parser.parse_args()

    # Handle verbose flag
    verbose = args.verbose and not args.quiet
    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    else:
        verbose = settings.DEFAULT_VERBOSE

    # Validate input file
    if not Path(args.document_path).exists():
        print(f"❌ File not found: {args.document_path}")
        return 1

    print("🚀 Starting Graph RAG Training")
    print("=" * 50)
    print(f"📄 Document: {args.document_path}")
    print(f"📊 Graph Name: {args.graph_name or settings.DEFAULT_GRAPH_NAME}")
    print(f"💾 Storage Directory: {args.storage_dir or settings.DEFAULT_STORAGE_DIR}")
    print("=" * 50)

    try:
        # Initialize trainer
        trainer = GraphRAGTrainer(
            use_neo4j=args.neo4j,
            verbose=verbose,
            storage_dir=args.storage_dir
        )

        # Train and save
        storage_path = trainer.train_from_file(args.document_path, args.graph_name)

        print(f"\n✅ Training completed successfully!")
        print(f"📍 Knowledge graph saved to: {storage_path}")
        print(f"\n💡 To query this graph, run:")
        graph_name = args.graph_name or settings.DEFAULT_GRAPH_NAME
        print(f"   python query.py --graph-name {graph_name}")

        return 0

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())