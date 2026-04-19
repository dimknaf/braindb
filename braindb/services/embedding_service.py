"""
Embedding service — generates and compares keyword embeddings.
Uses Qwen/Qwen3-Embedding-0.6B (384-dim) via sentence-transformers.
Adapted from fa-automation's embedding stack.

The model lazy-loads on first use. If sentence-transformers isn't installed
or the model fails to load, all methods return None and the system runs text-only.
"""
import logging

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024


class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self._initialized = False
        self._available = False

    def initialize(self) -> bool:
        """Lazy-load the embedding model. Returns True if ready."""
        if self._initialized:
            return self._available
        self._initialized = True
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s ...", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            self._available = True
            logger.info("Embedding model loaded: %s (%d-dim)", self.model_name, EMBEDDING_DIM)
            return True
        except ImportError:
            logger.warning("sentence-transformers not installed — embeddings disabled")
            return False
        except Exception as e:
            logger.warning("Failed to load embedding model: %s", e)
            return False

    def is_available(self) -> bool:
        return self._available

    def embed(self, text: str) -> list[float] | None:
        """Embed a single text string. Returns list of floats or None."""
        if not self.initialize():
            return None
        try:
            return self.model.encode([text], show_progress_bar=False)[0].tolist()
        except Exception as e:
            logger.warning("Embedding failed for text '%s': %s", text[:50], e)
            return None

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]] | None:
        """Embed multiple texts. Returns list of embedding vectors or None."""
        if not self.initialize() or not texts:
            return None
        try:
            return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()
        except Exception as e:
            logger.warning("Batch embedding failed: %s", e)
            return None


# Module-level singleton — initialized once, reused across requests
_instance: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton EmbeddingService instance."""
    global _instance
    if _instance is None:
        _instance = EmbeddingService()
    return _instance
