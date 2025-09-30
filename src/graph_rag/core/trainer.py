import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.core import Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.indices import KnowledgeGraphIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

from ..config import settings

try:
    from llama_index.graph_stores.neo4j import Neo4jGraphStore
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class GraphRAGTrainer:
    def __init__(self, use_neo4j=False, verbose=None, storage_dir=None):
        self.verbose = verbose if verbose is not None else settings.DEFAULT_VERBOSE
        self.storage_dir = Path(storage_dir or settings.DEFAULT_STORAGE_DIR)
        self.storage_dir.mkdir(exist_ok=True)

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

        # Initialize graph store
        if use_neo4j and NEO4J_AVAILABLE and settings.neo4j_configured:
            self.graph_store = self._setup_neo4j()
        else:
            self.graph_store = SimpleGraphStore()
            if self.verbose:
                print("✅ Using SimpleGraphStore (in-memory)")

    def _setup_neo4j(self):
        """Setup Neo4j graph store"""
        try:
            graph_store = Neo4jGraphStore(
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USERNAME,
                password=settings.NEO4J_PASSWORD,
                database=settings.NEO4J_DATABASE
            )
            if self.verbose:
                print("✅ Using Neo4j graph store")
            return graph_store
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Neo4j not available ({e}), falling back to SimpleGraphStore")
            return SimpleGraphStore()

    def load_markdown_file(self, file_path: str) -> Optional[Document]:
        """Load and parse a markdown file into a Document object."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            document = Document(
                text=content,
                metadata={
                    "filename": Path(file_path).name,
                    "filepath": str(file_path),
                    "file_type": "markdown"
                }
            )
            if self.verbose:
                print(f"✅ Loaded markdown file: {file_path}")
                print(f"📄 Document Stats:")
                print(f"   - Length: {len(content):,} characters")
                print(f"   - Lines: {len(content.split(chr(10)))}")
                print(f"   - Words: {len(content.split())}")

            return document
        except Exception as e:
            print(f"❌ Error loading file {file_path}: {e}")
            return None

    def create_knowledge_graph(self, documents: List[Document]) -> KnowledgeGraphIndex:
        """Create a knowledge graph with structured training pipeline."""
        try:
            print("\n" + "=" * 80)
            print("🏭 TRAINING PHASE: KNOWLEDGE GRAPH CONSTRUCTION")
            print("=" * 80)

            # Configure node parser for chunking
            node_parser = SentenceSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )

            # Parse documents into chunks/nodes
            nodes = node_parser.get_nodes_from_documents(documents)

            print("\n📄 DOCUMENT INGESTION")
            print("=" * 40)
            print(f"Original document: {len(documents[0].text)} characters, {len(documents[0].text.split())} words")
            print(f"Chunking strategy: {settings.CHUNK_SIZE} tokens, {settings.CHUNK_OVERLAP} token overlap")
            print(f"Result: {len(nodes)} chunks created\n")

            # Display each chunk clearly
            if self.verbose:
                for i, node in enumerate(nodes):
                    print(f"CHUNK {i+1}")
                    print("=" * 20)
                    print(f"Length: {len(node.text)} chars, {len(node.text.split())} words")
                    print(f"Content: {node.text}")
                    print()

            print("🕸️ KNOWLEDGE GRAPH CREATION")
            print("=" * 40)

            # Configure storage context
            storage_context = StorageContext.from_defaults(graph_store=self.graph_store)

            # Create knowledge graph index
            print("🤖 Starting LLM-powered triplet extraction...\n")

            kg_index = KnowledgeGraphIndex(
                nodes=nodes,
                storage_context=storage_context,
                max_triplets_per_chunk=settings.MAX_TRIPLETS_PER_CHUNK,
                include_embeddings=True,
                show_progress=True
            )

            # Extract and analyze triplets
            self._analyze_triplets(kg_index, nodes)

            print("\n✅ TRAINING COMPLETE - Knowledge Graph Ready for Queries!")
            return kg_index

        except Exception as e:
            print(f"❌ Error creating knowledge graph: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            raise e

    def _analyze_triplets(self, kg_index: KnowledgeGraphIndex, nodes: List):
        """Analyze and display extracted triplets"""
        all_triplets = []
        all_entities = set()
        all_relations = set()
        triplets_found = False

        # Try multiple methods to extract triplets
        methods = [
            ("get_triplets", lambda: getattr(kg_index.graph_store, 'get_triplets', lambda: None)()),
            ("graph_dict", self._extract_from_graph_dict),
            ("networkx_graph", self._extract_from_networkx)
        ]

        for method_name, extractor in methods:
            if triplets_found:
                break
            try:
                if method_name == "get_triplets":
                    result = extractor()
                else:
                    result = extractor(kg_index)

                if result:
                    all_triplets = result
                    triplets_found = True
                    if self.verbose:
                        print(f"📊 Method {method_name}: Found {len(all_triplets)} triplets")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Method {method_name} failed: {e}")

        if all_triplets:
            # Collect unique entities and relations
            for triplet in all_triplets:
                all_entities.add(triplet[0])
                all_entities.add(triplet[2])
                all_relations.add(triplet[1])

            self._display_triplet_analysis(all_triplets, all_entities, all_relations, nodes)
        else:
            print("\n⚠️ No triplets found! This could indicate:")
            print("  • Graph store doesn't support triplet extraction")
            print("  • LLM failed to extract relationships")
            print("  • Different graph representation is being used")

    def _extract_from_graph_dict(self, kg_index):
        """Extract triplets from graph_dict attribute"""
        if hasattr(kg_index.graph_store, 'graph_dict'):
            graph_dict = kg_index.graph_store.graph_dict
            triplets = []
            for subj, pred_obj_dict in graph_dict.items():
                for pred, obj_list in pred_obj_dict.items():
                    for obj in obj_list:
                        triplets.append((subj, pred, obj))
            return triplets
        return None

    def _extract_from_networkx(self, kg_index):
        """Extract triplets from NetworkX graph"""
        if hasattr(kg_index, 'get_networkx_graph'):
            nx_graph = kg_index.get_networkx_graph()
            triplets = []
            for subj, obj, data in nx_graph.edges(data=True):
                pred = data.get('label', 'RELATED_TO')
                triplets.append((subj, pred, obj))
            return triplets
        return None

    def _display_triplet_analysis(self, all_triplets, all_entities, all_relations, nodes):
        """Display detailed analysis of extracted triplets"""
        triplets_per_chunk = max(1, len(all_triplets) // len(nodes)) if len(nodes) > 0 else 0

        print(f"\n📊 TRIPLET BREAKDOWN BY CHUNK:")
        print("-" * 50)

        for i in range(len(nodes)):
            start_idx = i * triplets_per_chunk
            end_idx = min(start_idx + triplets_per_chunk, len(all_triplets))
            chunk_triplets = all_triplets[start_idx:end_idx]

            print(f"\nCHUNK {i+1} TRIPLETS ({len(chunk_triplets)} total):")
            print("-" * 30)
            for j, triplet in enumerate(chunk_triplets):
                print(f"  {j+1}. ({triplet[0]}) --[{triplet[1]}]--> ({triplet[2]})")
            # if len(chunk_triplets) > 8:
            #     print(f"  ... and {len(chunk_triplets) - 8} more triplets")

        print("\n🗂️ TRAINING SUMMARY")
        print("=" * 40)
        print("Graph Store Type:", type(self.graph_store).__name__)
        print(f"Total Triplets: {len(all_triplets)}")
        print(f"Total Entities: {len(all_entities)}")
        print(f"Total Relations: {len(all_relations)}")
        print(f"Document Chunks: {len(nodes)} text chunks stored")
        print(f"Extracted Entities: {len(all_entities)} unique entities")
        print(f"Triplet Embeddings: {len(all_relations)} relationship embeddings")

    def save_knowledge_graph(self, kg_index: KnowledgeGraphIndex, name: str = None) -> str:
        """Save the knowledge graph to persistent storage"""
        try:
            graph_name = name or settings.DEFAULT_GRAPH_NAME
            # Create storage directory for this graph
            graph_dir = self.storage_dir / graph_name
            graph_dir.mkdir(exist_ok=True)

            # Persist the storage context
            kg_index.storage_context.persist(persist_dir=str(graph_dir))

            if self.verbose:
                print(f"✅ Knowledge graph saved to {graph_dir}")

            return str(graph_dir)
        except Exception as e:
            print(f"❌ Error saving knowledge graph: {e}")
            raise e

    def train_from_file(self, file_path: str, graph_name: str = None) -> str:
        """Complete training pipeline from file to saved graph"""
        # Load document
        document = self.load_markdown_file(file_path)
        if not document:
            raise ValueError(f"Failed to load document from {file_path}")

        # Create knowledge graph
        kg_index = self.create_knowledge_graph([document])

        # Save to storage
        storage_path = self.save_knowledge_graph(kg_index, graph_name)

        return storage_path