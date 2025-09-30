import time
from pathlib import Path
from typing import Optional

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices import KnowledgeGraphIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

from ..config import settings


class GraphRAGInference:
    def __init__(self, storage_dir=None, verbose=None):
        self.verbose = verbose if verbose is not None else settings.DEFAULT_VERBOSE
        self.storage_dir = Path(storage_dir or settings.DEFAULT_STORAGE_DIR)
        self.kg_index = None

        if self.verbose:
            print(f"🔑 Using OpenAI API key: {settings.OPENAI_API_KEY[:10]}...")

        # Configure LlamaIndex settings
        Settings.llm = OpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=settings.OPENAI_TEMPERATURE
        )
        Settings.embed_model = OpenAIEmbedding(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )

        if self.verbose:
            print(f"🤖 Configured LLM: {settings.OPENAI_MODEL}")
            print(f"🔢 Configured Embeddings: {settings.OPENAI_EMBEDDING_MODEL}")

    def load_knowledge_graph(self, graph_name: str = None) -> bool:
        """Load a stored knowledge graph"""
        try:
            graph_name = graph_name or settings.DEFAULT_GRAPH_NAME
            graph_dir = self.storage_dir / graph_name
            if not graph_dir.exists():
                print(f"❌ Knowledge graph '{graph_name}' not found at {graph_dir}")
                return False

            # Rebuild storage context from persisted data
            storage_context = StorageContext.from_defaults(persist_dir=str(graph_dir))

            # Load the knowledge graph index
            self.kg_index = load_index_from_storage(storage_context)

            if self.verbose:
                print(f"✅ Knowledge graph '{graph_name}' loaded from {graph_dir}")
                print(f"📊 Graph type: {type(self.kg_index).__name__}")

            return True

        except Exception as e:
            print(f"❌ Error loading knowledge graph: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def list_available_graphs(self) -> list:
        """List all available knowledge graphs in storage"""
        if not self.storage_dir.exists():
            return []

        graphs = []
        for item in self.storage_dir.iterdir():
            if item.is_dir() and (item / "docstore.json").exists():
                graphs.append(item.name)

        return graphs

    def query_graph(self, query_text: str) -> str:
        """Query the loaded knowledge graph with detailed inference pipeline."""
        if self.kg_index is None:
            return "❌ No knowledge graph loaded. Please load a graph first using load_knowledge_graph()"

        try:
            print("\n" + "=" * 80)
            print("🔍 INFERENCE PHASE: QUERY PROCESSING")
            print("=" * 80)

            print("\n📝 QUERY")
            print("=" * 20)
            print(f"User Query: '{query_text}'")

            print("\n🔢 EMBEDDING GENERATION")
            print("=" * 30)
            print("🔄 Converting query to embedding vector...")
            query_embedding = Settings.embed_model.get_text_embedding(query_text)
            print(f"✅ Query embedded to {len(query_embedding)}-dimensional vector")
            if self.verbose:
                print(f"🔢 First 5 dimensions: {query_embedding[:5]}")

            print("\n🎯 HYBRID RETRIEVAL PROCESS")
            print("=" * 30)
            print("1. Keyword extraction from query")
            print("2. Graph traversal via keyword → node lookup")
            print("3. Triplet embedding similarity search")
            print("4. Hybrid combination and frequency-based ranking")

            # Perform retrieval to see what gets retrieved
            retriever = self.kg_index.as_retriever(similarity_top_k=settings.SIMILARITY_TOP_K)
            retrieved_nodes = retriever.retrieve(query_text)

            print(f"\n📋 RETRIEVED CONTEXT")
            print("=" * 30)
            print(f"Retrieved {len(retrieved_nodes)} relevant nodes:")

            context_pieces = []
            for i, node_with_score in enumerate(retrieved_nodes):
                print(f"\n  Node {i+1} (Similarity Score: {node_with_score.score:.3f}):")
                print(f"  Length: {len(node_with_score.node.text)} chars")
                print(f"  Content: {node_with_score.node.text[:150]}...")
                context_pieces.append(node_with_score.node.text)

            # Assemble full context
            full_context = "\n\n".join(context_pieces)
            print(f"\nTotal Context Length: {len(full_context)} characters")

            print("\n📤 LLM PROMPT CONSTRUCTION")
            print("=" * 30)

            # Approximate the prompt structure
            system_prompt = "You are a helpful assistant that answers questions based on the provided context."
            context_section = f"Context information:\n{full_context[:500]}..."
            query_section = f"Query: {query_text}"
            instruction = "Answer the query based only on the context provided above."

            full_prompt = f"{system_prompt}\n\n{context_section}\n\n{query_section}\n\n{instruction}"

            if self.verbose:
                print(f"System Prompt: {system_prompt}")
                print(f"Context Length: {len(full_context)} chars")
                print(f"Context Preview: {full_context[:200]}...")
                print(f"Query: {query_text}")
                print(f"Total Prompt Length: {len(full_prompt)} chars")

            print("\n⏳ LLM PROCESSING")
            print("=" * 30)

            # Create query engine and get response
            query_engine = self.kg_index.as_query_engine(
                include_text=True,
                response_mode=settings.RESPONSE_MODE
            )

            start_time = time.time()
            response = query_engine.query(query_text)
            end_time = time.time()

            print(f"⏱️ LLM Processing Time: {end_time - start_time:.2f} seconds")

            print("\n🎯 FINAL RESPONSE")
            print("=" * 30)
            print(f"Response Length: {len(str(response))} characters")
            print(f"Response Type: {type(response)}")
            print("\nResponse Content:")
            print("-" * 20)
            print(str(response))

            print("\n" + "=" * 80)
            print("✅ INFERENCE COMPLETE")
            print("=" * 80)

            return str(response)

        except Exception as e:
            print(f"❌ Error querying graph: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return f"❌ Error querying graph: {e}"

    def get_graph_stats(self) -> dict:
        """Get statistics about the loaded knowledge graph"""
        if self.kg_index is None:
            return {"error": "No knowledge graph loaded"}

        try:
            stats = {
                "graph_type": type(self.kg_index).__name__,
                "graph_store_type": type(self.kg_index.graph_store).__name__,
                "has_embeddings": hasattr(self.kg_index, "_embed_model") and self.kg_index._embed_model is not None
            }

            # Try to get more detailed stats if available
            try:
                if hasattr(self.kg_index.graph_store, 'get_triplets'):
                    triplets = self.kg_index.graph_store.get_triplets()
                    stats["num_triplets"] = len(triplets)

                    # Count unique entities and relations
                    entities = set()
                    relations = set()
                    for triplet in triplets:
                        entities.add(triplet[0])
                        entities.add(triplet[2])
                        relations.add(triplet[1])

                    stats["num_entities"] = len(entities)
                    stats["num_relations"] = len(relations)

            except Exception:
                stats["triplet_stats"] = "Not available"

            return stats

        except Exception as e:
            return {"error": f"Failed to get stats: {e}"}

    def interactive_query_session(self):
        """Start an interactive query session"""
        if self.kg_index is None:
            print("❌ No knowledge graph loaded. Please load a graph first.")
            return

        print("\n🎯 Interactive Query Session")
        print("=" * 50)
        print("Type your queries below. Type 'quit' to exit.")
        print("=" * 50)

        while True:
            try:
                query = input("\n🔍 Query: ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if not query:
                    continue

                response = self.query_graph(query)
                print(f"\n💡 Answer: {response}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def query_simple(self, query_text: str) -> str:
        """Simple query without detailed logging for programmatic use"""
        if self.kg_index is None:
            return "❌ No knowledge graph loaded."

        try:
            query_engine = self.kg_index.as_query_engine(
                include_text=True,
                response_mode=settings.RESPONSE_MODE
            )
            response = query_engine.query(query_text)
            return str(response)

        except Exception as e:
            return f"❌ Error querying graph: {e}"