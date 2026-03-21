# ABOUTME: Tests for chunking module - sentence-based and paragraph-aware strategies.
# ABOUTME: Covers the chunk_text, create_chunks_from_pages, and paragraph chunking functions.

import pytest
from pdfrag.chunking import (
    chunk_text,
    create_chunks_from_pages,
    is_section_header,
    chunk_by_paragraphs,
    create_paragraph_chunks_from_pages,
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


# --- is_section_header ---

def test_is_section_header_numbered():
    assert is_section_header("3.2.1 Overview of the System") is True

def test_is_section_header_single_level():
    assert is_section_header("1. Introduction") is True

def test_is_section_header_chapter():
    assert is_section_header("CHAPTER 5") is True

def test_is_section_header_section():
    assert is_section_header("SECTION 3") is True

def test_is_section_header_appendix():
    assert is_section_header("APPENDIX A") is True

def test_is_section_header_plain_text():
    assert is_section_header("This is a normal sentence, not a header.") is False

def test_is_section_header_number_only():
    # No title text after the number - not a section header
    assert is_section_header("3.2.1") is False


# --- chunk_by_paragraphs ---

def test_chunk_by_paragraphs_merges_short_paragraphs():
    # Two short paragraphs well under chunk_chars limit - should merge into one chunk
    text = "Short paragraph one.\n\nShort paragraph two."
    chunks = chunk_by_paragraphs(text, chunk_chars=2500)
    assert len(chunks) == 1
    assert "Short paragraph one" in chunks[0]
    assert "Short paragraph two" in chunks[0]


def test_chunk_by_paragraphs_splits_when_over_limit():
    # Each paragraph is 60 chars; limit is 80 - first fits, second would overflow
    para = "A" * 60
    text = "\n\n".join([para] * 4)
    chunks = chunk_by_paragraphs(text, chunk_chars=80)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 80


def test_chunk_by_paragraphs_section_header_forces_break():
    # Both parts fit in 2500 chars, but section header should force a split
    text = "Some introductory text before the section.\n\n3.2.1 New Section Title\nContent of the new section."
    chunks = chunk_by_paragraphs(text, chunk_chars=2500)
    assert len(chunks) == 2
    assert "3.2.1 New Section Title" in chunks[1]


def test_chunk_by_paragraphs_oversized_split_with_overlap():
    # Single paragraph larger than chunk_chars - must be split with overlap
    big_para = "X" * 600
    chunks = chunk_by_paragraphs(big_para, chunk_chars=200, overlap_chars=50)
    assert len(chunks) > 1
    # Verify overlap: end of chunk 0 equals start of chunk 1
    assert chunks[0][-50:] == chunks[1][:50]


def test_chunk_by_paragraphs_skips_empty_paragraphs():
    text = "First paragraph.\n\n\n\n\n\nSecond paragraph."
    chunks = chunk_by_paragraphs(text, chunk_chars=2500)
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_by_paragraphs_empty_text():
    assert chunk_by_paragraphs("") == []


# --- create_paragraph_chunks_from_pages ---

def test_create_paragraph_chunks_from_pages_metadata():
    pages = [
        {"page": 1, "text": "First page content.\n\nMore content on page one."},
        {"page": 2, "text": "Second page content."},
    ]
    chunks = create_paragraph_chunks_from_pages(pages)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "page" in chunk
        assert "chunk_index" in chunk


def test_create_paragraph_chunks_from_pages_page_numbers():
    pages = [
        {"page": 3, "text": "Page three text."},
        {"page": 7, "text": "Page seven text."},
    ]
    chunks = create_paragraph_chunks_from_pages(pages)
    page_nums = [c["page"] for c in chunks]
    assert 3 in page_nums
    assert 7 in page_nums


def test_create_paragraph_chunks_from_pages_monotonic_indices():
    pages = [
        {"page": 1, "text": "Content.\n\nMore content.\n\nEven more content."},
        {"page": 2, "text": "Page two.\n\nPage two continued."},
    ]
    chunks = create_paragraph_chunks_from_pages(pages)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))
