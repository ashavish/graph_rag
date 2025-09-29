#!/usr/bin/env python3
"""
Query CLI for Graph RAG system.
Loads stored knowledge graphs and allows querying.
"""

import argparse
import sys
from pathlib import Path

from src.graph_rag.core import GraphRAGInference
from src.graph_rag.config import settings


def main():
    parser = argparse.ArgumentParser(description="Query stored Graph RAG knowledge graphs")
    parser.add_argument("--graph-name", "-n",
                       help=f"Name of the knowledge graph to load (default: '{settings.DEFAULT_GRAPH_NAME}')")
    parser.add_argument("--storage-dir", "-s",
                       help=f"Directory containing stored graphs (default: '{settings.DEFAULT_STORAGE_DIR}')")
    parser.add_argument("--query", "-q", help="Single query to ask")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Start interactive query session")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available knowledge graphs")
    parser.add_argument("--stats", action="store_true",
                       help="Show statistics about the loaded graph")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable detailed logging")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging")

    args = parser.parse_args()

    # Handle verbose flag
    verbose = args.verbose and not args.quiet
    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    else:
        verbose = settings.DEFAULT_VERBOSE

    try:
        # Initialize inference engine
        inference = GraphRAGInference(
            storage_dir=args.storage_dir,
            verbose=verbose
        )

        # List available graphs
        if args.list:
            graphs = inference.list_available_graphs()
            if graphs:
                print("📊 Available Knowledge Graphs:")
                print("=" * 40)
                for graph in graphs:
                    print(f"  • {graph}")
                print("\n💡 Use --graph-name <name> to select a graph")
            else:
                print("❌ No knowledge graphs found in storage directory")
                print("💡 Train a graph first using: python train.py <document>")
            return 0

        # Load the specified graph
        graph_name = args.graph_name or settings.DEFAULT_GRAPH_NAME
        print("🔍 Loading Knowledge Graph")
        print("=" * 40)
        if not inference.load_knowledge_graph(graph_name):
            print(f"❌ Failed to load graph '{graph_name}'")
            print(f"💡 Available graphs: {inference.list_available_graphs()}")
            return 1

        # Show graph statistics
        if args.stats:
            stats = inference.get_graph_stats()
            print(f"\n📊 Graph Statistics for '{graph_name}':")
            print("=" * 50)
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return 0

        # Single query mode
        if args.query:
            print(f"\n🔍 Querying: '{args.query}'")
            response = inference.query_graph(args.query)
            return 0

        # Interactive mode
        if args.interactive:
            inference.interactive_query_session()
            return 0

        # Default: show help if no action specified
        print("\n💡 No action specified. Choose one of:")
        print("  --list          : List available graphs")
        print("  --query 'text'  : Ask a single question")
        print("  --interactive   : Start interactive session")
        print("  --stats         : Show graph statistics")
        print(f"\nExample:")
        print(f"  python query.py --graph-name {graph_name} --interactive")

        return 0

    except Exception as e:
        print(f"❌ Query failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())