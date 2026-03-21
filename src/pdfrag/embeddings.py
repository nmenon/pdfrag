# ABOUTME: Generates text embeddings using sentence-transformers models.
# ABOUTME: Wraps SentenceTransformer for consistent embedding generation interface.

"""Text embedding generation using sentence-transformers."""

from typing import List, Optional
from sentence_transformers import SentenceTransformer

# Default embedding model
DEFAULT_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


class EmbeddingGenerator:
    """Generates embeddings using sentence-transformers.

    Wraps SentenceTransformer model for generating 768-dimensional embeddings
    optimized for question-answering and semantic search tasks.

    Attributes:
        model: SentenceTransformer model instance
        model_name: Name of the loaded model
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        """Initialize embedding generator with specified model.

        Args:
            model_name: Name of sentence-transformers model to use
                       (default: multi-qa-mpnet-base-dot-v1)
            device: Torch device to run inference on ('cpu', 'cuda', etc.).
                   None lets sentence-transformers auto-detect (prefers GPU).

        Example:
            >>> generator = EmbeddingGenerator()
            >>> embeddings = generator.generate(["Hello world"])
            >>> print(len(embeddings[0]))  # 768
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def generate(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar during encoding

        Returns:
            List of embedding vectors (each 768 dimensions)

        Example:
            >>> generator = EmbeddingGenerator()
            >>> texts = ["First text", "Second text"]
            >>> embeddings = generator.generate(texts)
            >>> print(len(embeddings))  # 2
            >>> print(len(embeddings[0]))  # 768
        """
        embeddings = self.model.encode(texts, show_progress_bar=show_progress)
        return embeddings.tolist()

    def generate_single(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector (768 dimensions)

        Example:
            >>> generator = EmbeddingGenerator()
            >>> embedding = generator.generate_single("Hello world")
            >>> print(len(embedding))  # 768
        """
        embedding = self.model.encode([text], show_progress_bar=False)[0]
        return embedding.tolist()
