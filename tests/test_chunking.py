# ABOUTME: Tests for chunking module - sentence-based and paragraph-aware strategies.
# ABOUTME: Covers the chunk_text, create_chunks_from_pages, and paragraph chunking functions.

import pytest
from pdfrag.chunking import (
    chunk_text,
    create_chunks_from_pages,
)


# --- Regression: chunk_text name-shadowing bug ---

def test_create_chunks_from_pages_multipage():
    """create_chunks_from_pages must not raise UnboundLocalError on any page."""
    pages = [
        {"page": 1, "text": "First sentence. Second sentence. Third sentence."},
        {"page": 2, "text": "Fourth sentence. Fifth sentence. Sixth sentence."},
    ]
    chunks = create_chunks_from_pages(pages, chunk_size=2, overlap=1)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "page" in chunk
        assert "chunk_index" in chunk


def test_create_chunks_from_pages_single_page():
    pages = [{"page": 1, "text": "One sentence. Two sentences. Three sentences."}]
    chunks = create_chunks_from_pages(pages, chunk_size=2, overlap=1)
    assert len(chunks) > 0
    assert chunks[0]["page"] == 1


def test_chunk_indices_are_monotonic():
    pages = [
        {"page": 1, "text": "A. B. C. D. E."},
        {"page": 2, "text": "F. G. H. I. J."},
    ]
    chunks = create_chunks_from_pages(pages, chunk_size=2, overlap=0)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


# --- chunk_text unit tests ---

def test_chunk_text_basic():
    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
    chunks = chunk_text(text, chunk_size=2, overlap=0)
    assert len(chunks) == 2


def test_chunk_text_with_overlap():
    # Use full sentences so NLTK tokenizes them correctly.
    text = "The cat sat. The dog ran. The bird flew. The fish swam."
    chunks = chunk_text(text, chunk_size=2, overlap=1)
    # 4 sentences, chunk_size=2, overlap=1: step=1
    # [s0,s1], [s1,s2], [s2,s3], [s3] = 4 chunks
    assert len(chunks) == 4
    # s1 appears in both first and second chunk
    assert "dog" in chunks[0]
    assert "dog" in chunks[1]


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=3, overlap=1) == []
