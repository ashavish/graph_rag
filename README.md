# Graph RAG System

A modular Graph-based Retrieval-Augmented Generation (RAG) system with separate training and inference pipelines. This repo uses Llama Index to demo a simple Graph RAG pipeline.

## Overview

This system creates knowledge graphs from markdown files and allows you to query them using natural language. The training and inference phases are completely separated, allowing you to:

1. **Train once**: Build knowledge graphs from your markdowns
2. **Query many times**: Ask questions using stored graphs without reprocessing

## Architecture

```
src/graph_rag/
├── core/                    # Core functionality
│   ├── trainer.py          # Training module for creating knowledge graphs
│   └── inference.py        # Inference module for querying stored graphs
├── config/                  # Configuration management
│   └── settings.py         # Environment-based settings
├── storage/                 # Storage utilities (future)
└── utils/                   # Utility functions (future)

# Root-level CLI scripts
train.py                     # Training CLI - direct implementation
query.py                     # Query CLI - direct implementation
```

## Prerequisites

- Python 3.9+
- UV package manager
- OpenAI API key (required)
- Neo4j database (optional, for production)

## Installation

1. Clone or navigate to the project directory
2. Install dependencies with UV:
   ```bash
   uv sync
   ```

## Configuration

⚠️ **Security Note**: This project uses environment variables for configuration. Never commit API keys to version control.

### Setup Environment Variables

1. **Required**: Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

2. **Optional**: Copy and customize the example environment file:
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings (optional)
   ```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional OpenAI settings
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.1
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Optional: Neo4j configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# Document processing
CHUNK_SIZE=256
CHUNK_OVERLAP=50
MAX_TRIPLETS_PER_CHUNK=10

# Storage
GRAPH_STORAGE_DIR=./storage
DEFAULT_GRAPH_NAME=default
```

## Quick Start

### 1. Train a Knowledge Graph

```bash
# Train from a document
python train.py sample_document.md --graph-name my_graph

# With custom storage location
python train.py sample_document.md --graph-name my_graph --storage-dir ./my_storage
```

### 2. Query the Knowledge Graph

```bash
# Interactive mode
python query.py --graph-name my_graph --interactive

# Single query
python query.py --graph-name my_graph --query "What are the main topics?"

# List available graphs
python query.py --list
```

## Detailed Usage

### Training Pipeline

```bash
python train.py document.md [options]

Options:
  --graph-name, -n      Name for the knowledge graph (default: 'default')
  --storage-dir, -s     Storage directory (default: './storage')
  --neo4j              Use Neo4j graph store (requires Neo4j setup)
  --verbose, -v        Enable detailed logging
  --quiet, -q          Disable verbose logging
```

### Inference Pipeline

```bash
python query.py [options]

Options:
  --graph-name, -n     Graph to load (default: 'default')
  --storage-dir, -s    Storage directory (default: './storage')
  --query, -q          Single query to ask
  --interactive, -i    Start interactive session
  --list, -l          List available graphs
  --stats             Show graph statistics
  --verbose, -v       Enable detailed logging
```

## Programmatic Usage

```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

from src.graph_rag import GraphRAGTrainer, GraphRAGInference

# Training
trainer = GraphRAGTrainer()
storage_path = trainer.train_from_file("document.md", "my_graph")

# Inference
inference = GraphRAGInference()
inference.load_knowledge_graph("my_graph")
response = inference.query_simple("What are the main topics?")
```

## Examples

### Training from Multiple Documents

```bash
# Train individual graphs
python train.py doc1.md --graph-name doc1_graph
python train.py doc2.md --graph-name doc2_graph
```

### Querying Different Graphs

```bash
# Compare responses from different graphs
python query.py --graph-name doc1_graph --query "What is the main concept?"
python query.py --graph-name doc2_graph --query "What is the main concept?"
```

### Interactive Session

```bash
python query.py --graph-name my_graph --interactive

# Then interactively:
# 🔍 Query: What are the key concepts?
# 🔍 Query: How do these concepts relate?
# 🔍 Query: quit
```

## Storage Structure

Knowledge graphs are stored in the following structure:

```
storage/
├── graph_name_1/
│   ├── docstore.json          # Document storage
│   ├── graph_store.json       # Graph structure
│   ├── index_store.json       # Index mappings
│   └── vector_store.json      # Vector embeddings
└── graph_name_2/
    └── ...
```

## Features

- ✅ **Modular Architecture**: Separate training and inference
- ✅ **Persistent Storage**: Save and load knowledge graphs
- ✅ **Multiple Graph Support**: Manage multiple knowledge graphs
- ✅ **Interactive Queries**: Chat-like interface for questions
- ✅ **Detailed Logging**: Understand the RAG process
- ✅ **Neo4j Support**: Scale with graph databases
- ✅ **Environment-based Config**: Secure configuration management
- ✅ **Graph Statistics**: Analyze your knowledge graphs

## Security Best Practices

- 🔐 **Never commit API keys**: Use environment variables only
- 🔐 **Use .env.example**: Provide template without secrets
- 🔐 **Validate configuration**: Settings module validates API keys
- 🔐 **Secure storage**: Knowledge graphs stored locally by default

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   Error: Required environment variable 'OPENAI_API_KEY' is not set
   Solution: export OPENAI_API_KEY=your_actual_key
   ```

2. **Graph Not Found**
   ```
   Error: Knowledge graph 'my_graph' not found
   Solution: Use --list to see available graphs or train first
   ```

3. **Invalid API Key Format**
   ```
   Error: OPENAI_API_KEY does not appear to be a valid OpenAI API key
   Solution: Ensure key starts with 'sk-' or 'sk-proj-'
   ```

### Debug Mode

Use `--verbose` flag for detailed logging:

```bash
python train.py document.md --verbose
python query.py --graph-name my_graph --query "test" --verbose
```

## Dependencies

Core dependencies:
- `llama-index`: Core LlamaIndex functionality
- `llama-index-graph-stores-neo4j`: Neo4j integration (optional)
- `llama-index-embeddings-openai`: OpenAI embeddings
- `llama-index-llms-openai`: OpenAI language models
- `networkx`: Graph manipulation and analysis
- `matplotlib`: Graph visualization (optional)

## Contributing

This project emphasizes security and modularity. When contributing:
- Never commit API keys or secrets
- Use environment variables for all configuration
- Follow the modular architecture pattern
- Add proper error handling and validation

## License

This project is for educational and demonstration purposes.