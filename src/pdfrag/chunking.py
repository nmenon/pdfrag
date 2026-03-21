# ABOUTME: Performs semantic chunking by grouping sentences with configurable overlap.
# ABOUTME: Uses NLTK sentence tokenization to preserve sentence boundaries.

"""Semantic text chunking with sentence boundaries."""

import re
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

        for chunk in chunks:
            all_chunks.append({
                'text': chunk,
                'page': page_num,
                'chunk_index': global_chunk_idx
            })
            global_chunk_idx += 1

    return all_chunks


# Patterns that indicate a section header line
_HEADER_PATTERNS = [
    re.compile(r'^\d+(\.\d+)+\s+\S'),               # 3.2.1 Title, 1.2 Title
    re.compile(r'^\d+\.\s+\S'),                       # 1. Title
    re.compile(r'^(CHAPTER|SECTION|APPENDIX)\s+\w+', re.IGNORECASE),  # CHAPTER 5
]


def is_section_header(text: str) -> bool:
    """Return True if text looks like a section header.

    Recognises numbered headers (1. Intro, 3.2.1 Overview) and keyword
    headers (CHAPTER 5, SECTION 3, APPENDIX A).

    Args:
        text: Text to test (typically a paragraph or line)

    Returns:
        True when the text matches a known header pattern
    """
    line = text.strip()
    return any(p.match(line) for p in _HEADER_PATTERNS)


def chunk_by_paragraphs(text: str, chunk_chars: int = 2500,
                        overlap_chars: int = 200) -> List[str]:
    """Chunk text using paragraph boundaries.

    Splits on blank lines, merges short paragraphs up to chunk_chars,
    forces a break before section headers, and splits oversized paragraphs
    with character-level overlap.

    Args:
        text: Text to chunk
        chunk_chars: Maximum characters per chunk (default: 2500)
        overlap_chars: Characters of overlap when splitting oversized paragraphs (default: 200)

    Returns:
        List of text chunks
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r'\n\n+', text)]
    paragraphs = [p for p in paragraphs if p]

    chunks = []
    current = ''

    for para in paragraphs:
        # Force a break before section headers when we already have content
        if is_section_header(para) and current:
            chunks.append(current)
            current = para
            continue

        # Oversized paragraph: flush current, then split with overlap
        if len(para) > chunk_chars:
            if current:
                chunks.append(current)
                current = ''
            i = 0
            while i < len(para):
                end = min(i + chunk_chars, len(para))
                chunks.append(para[i:end])
                if end == len(para):
                    break
                i = end - overlap_chars
            continue

        # Merge with current chunk or start a new one
        sep = '\n\n' if current else ''
        if current and len(current) + len(sep) + len(para) > chunk_chars:
            chunks.append(current)
            current = para
        else:
            current = current + sep + para if current else para

    if current:
        chunks.append(current)

    return chunks


def create_paragraph_chunks_from_pages(pages_text: List[Dict[str, Any]],
                                       chunk_chars: int = 2500,
                                       overlap_chars: int = 200) -> List[Dict[str, Any]]:
    """Create paragraph-aware chunks from PDF pages with metadata.

    Args:
        pages_text: List of page dicts with 'page' and 'text'
        chunk_chars: Maximum characters per chunk (default: 2500)
        overlap_chars: Character overlap for oversized paragraphs (default: 200)

    Returns:
        List of chunk dicts with text, page number, and chunk index
    """
    all_chunks = []
    global_chunk_idx = 0

    for page_data in pages_text:
        page_num = page_data['page']
        page_text = page_data['text']

        chunks = chunk_by_paragraphs(page_text, chunk_chars, overlap_chars)

        for chunk in chunks:
            all_chunks.append({
                'text': chunk,
                'page': page_num,
                'chunk_index': global_chunk_idx
            })
            global_chunk_idx += 1

    return all_chunks
