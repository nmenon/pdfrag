# ABOUTME: Performs semantic chunking by grouping sentences with configurable overlap.
# ABOUTME: Uses NLTK sentence tokenization to preserve sentence boundaries.

"""Semantic text chunking with sentence boundaries."""

from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 3  # Number of sentences per chunk
DEFAULT_OVERLAP = 1  # Sentence overlap between chunks

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """Perform semantic chunking by splitting text into sentences and grouping them.

    Splits text at sentence boundaries and groups consecutive sentences into chunks
    with configurable overlap. Preserves context by overlapping sentences between
    adjacent chunks.

    Args:
        text: Text to chunk
        chunk_size: Number of sentences per chunk (default: 3)
        overlap: Number of sentences to overlap between chunks (default: 1)

    Returns:
        List of text chunks

    Example:
        >>> text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        >>> chunks = chunk_text(text, chunk_size=2, overlap=1)
        >>> print(len(chunks))  # 3 chunks with overlap
    """
    # Split into sentences
    sentences = sent_tokenize(text)

    if not sentences:
        return []

    chunks = []
    i = 0

    while i < len(sentences):
        # Take chunk_size sentences
        chunk_sentences = sentences[i:i + chunk_size]
        chunk = ' '.join(chunk_sentences)
        chunks.append(chunk)

        # Move forward by (chunk_size - overlap) to create overlap
        i += max(1, chunk_size - overlap)

    return chunks


def create_chunks_from_pages(pages_text: List[Dict[str, Any]],
                             chunk_size: int = DEFAULT_CHUNK_SIZE,
                             overlap: int = DEFAULT_OVERLAP) -> List[Dict[str, Any]]:
    """Create semantic chunks from PDF pages with metadata.

    Processes each page's text through semantic chunking and attaches metadata
    including page numbers and chunk indices.

    Args:
        pages_text: List of page dictionaries with 'page' and 'text'
        chunk_size: Number of sentences per chunk (default: 3)
        overlap: Sentence overlap between chunks (default: 1)

    Returns:
        List of chunk dictionaries with text, page number, and chunk index

    Example:
        >>> pages = [{"page": 1, "text": "Content..."}]
        >>> chunks = create_chunks_from_pages(pages)
        >>> print(chunks[0]["page"])  # 1
    """
    all_chunks = []
    global_chunk_idx = 0

    for page_data in pages_text:
        page_num = page_data['page']
        page_text = page_data['text']

        # Create chunks for this page
        chunks = chunk_text(page_text, chunk_size, overlap)

        for chunk_text in chunks:
            all_chunks.append({
                'text': chunk_text,
                'page': page_num,
                'chunk_index': global_chunk_idx
            })
            global_chunk_idx += 1

    return all_chunks
