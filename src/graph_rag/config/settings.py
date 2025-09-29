"""
Configuration settings for Graph RAG system.
All sensitive information should be provided via environment variables.
"""

import os
from pathlib import Path
from typing import Optional


class Settings:
    """Configuration class for Graph RAG system using environment variables"""

    def __init__(self):
        # OpenAI Settings - Required
        self.OPENAI_API_KEY = self._get_required_env("OPENAI_API_KEY")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        self.OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        # Neo4j Settings (optional)
        self.NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        self.NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

        # Document Processing Settings
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.MAX_TRIPLETS_PER_CHUNK = int(os.getenv("MAX_TRIPLETS_PER_CHUNK", "10"))

        # Storage Settings
        self.DEFAULT_STORAGE_DIR = os.getenv("GRAPH_STORAGE_DIR", "./storage")
        self.DEFAULT_GRAPH_NAME = os.getenv("DEFAULT_GRAPH_NAME", "default")

        # Retrieval Settings
        self.SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))
        self.RESPONSE_MODE = os.getenv("RESPONSE_MODE", "tree_summarize")

        # Logging Settings
        self.DEFAULT_VERBOSE = os.getenv("DEFAULT_VERBOSE", "true").lower() == "true"
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

        # Validate configuration
        self.validate()

    def _get_required_env(self, key: str) -> str:
        """Get required environment variable or raise error"""
        value = os.getenv(key)
        if not value:
            raise ValueError(
                f"Required environment variable '{key}' is not set. "
                f"Please set it before running the application."
            )
        return value

    def validate(self) -> bool:
        """Validate essential configuration"""
        errors = []

        # Validate OpenAI API key format (basic check)
        if not self.OPENAI_API_KEY.startswith(('sk-', 'sk-proj-')):
            errors.append("OPENAI_API_KEY does not appear to be a valid OpenAI API key")

        # Validate temperature range
        if not 0.0 <= self.OPENAI_TEMPERATURE <= 2.0:
            errors.append("OPENAI_TEMPERATURE must be between 0.0 and 2.0")

        # Validate chunk settings
        if self.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE must be positive")

        if self.CHUNK_OVERLAP < 0:
            errors.append("CHUNK_OVERLAP cannot be negative")

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")

        # Create storage directory if it doesn't exist
        storage_path = Path(self.DEFAULT_STORAGE_DIR)
        try:
            storage_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create storage directory {storage_path}: {e}")

        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

        return True

    @property
    def neo4j_configured(self) -> bool:
        """Check if Neo4j is properly configured"""
        return bool(self.NEO4J_PASSWORD)

    def get_summary(self) -> dict:
        """Get a summary of current configuration (without sensitive data)"""
        return {
            "openai_model": self.OPENAI_MODEL,
            "openai_temperature": self.OPENAI_TEMPERATURE,
            "embedding_model": self.OPENAI_EMBEDDING_MODEL,
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "max_triplets_per_chunk": self.MAX_TRIPLETS_PER_CHUNK,
            "storage_dir": self.DEFAULT_STORAGE_DIR,
            "similarity_top_k": self.SIMILARITY_TOP_K,
            "response_mode": self.RESPONSE_MODE,
            "neo4j_configured": self.neo4j_configured,
            "api_key_configured": bool(self.OPENAI_API_KEY),
            "log_level": self.LOG_LEVEL
        }

    def __repr__(self) -> str:
        """String representation without sensitive data"""
        return f"Settings(model={self.OPENAI_MODEL}, storage={self.DEFAULT_STORAGE_DIR})"


# Global settings instance
settings = Settings()