# ABOUTME: Tests for Pydantic input models in the MCP server.
# ABOUTME: Verifies that new parameters (rerank, chunking_strategy) are accepted correctly.

import pytest
from pdfrag.server import PdfSearchInput, PdfAddInput, ChunkingStrategy


def test_pdf_search_input_rerank_defaults_false():
    params = PdfSearchInput(query="what is a transistor")
    assert params.rerank is False


def test_pdf_search_input_rerank_true():
    params = PdfSearchInput(query="what is a transistor", rerank=True)
    assert params.rerank is True


def test_pdf_add_input_chunking_strategy_defaults_sentence(tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.touch()
    params = PdfAddInput(pdf_path=str(pdf_file))
    assert params.chunking_strategy == ChunkingStrategy.SENTENCE


def test_pdf_add_input_chunking_strategy_paragraph(tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.touch()
    params = PdfAddInput(pdf_path=str(pdf_file), chunking_strategy="paragraph")
    assert params.chunking_strategy == ChunkingStrategy.PARAGRAPH
