#!/usr/bin/env python3
"""
Example usage script for PDF RAG MCP Server

This script demonstrates how the MCP server processes PDFs.
It can be used for testing before integrating with Claude Desktop.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_rag_mcp import (
    extract_text_from_pdf,
    semantic_chunking,
    create_chunks_from_pdf,
    get_file_hash
)


def test_pdf_processing(pdf_path: str):
    """
    Test PDF processing functions without MCP server.

    Args:
        pdf_path: Path to a test PDF file
    """
    print(f"\n{'='*60}")
    print(f"Testing PDF Processing: {Path(pdf_path).name}")
    print(f"{'='*60}\n")

    # 1. Extract text
    print("Step 1: Extracting text from PDF...")
    try:
        pages_text = extract_text_from_pdf(pdf_path)
        print(f"✅ Extracted text from {len(pages_text)} pages")

        # Check if OCR was used on any pages
        ocr_pages = [p for p in pages_text if p.get('ocr_used', False)]
        if ocr_pages:
            print(f"📄 OCR was used on {len(ocr_pages)} page(s)")

        # Show first page preview
        if pages_text:
            first_page = pages_text[0]
            preview = first_page['text'][:200] + "..." if len(first_page['text']) > 200 else first_page['text']
            print(f"\nFirst page preview:\n{preview}\n")
            if first_page.get('ocr_used'):
                print("  (extracted via OCR)")
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return
    
    # 2. Create chunks
    print("Step 2: Creating semantic chunks...")
    try:
        chunks = create_chunks_from_pdf(pages_text, chunk_size=3, overlap=1)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Show first chunk
        if chunks:
            first_chunk = chunks[0]
            print(f"\nFirst chunk (Page {first_chunk['page']}):")
            print(f"{first_chunk['text'][:300]}...")
            print(f"\nChunk Statistics:")
            print(f"  - Total chunks: {len(chunks)}")
            print(f"  - Average chunk length: {sum(len(c['text']) for c in chunks) / len(chunks):.0f} characters")
    except Exception as e:
        print(f"❌ Error creating chunks: {e}")
        return
    
    # 3. Generate document ID
    print("\nStep 3: Generating document ID...")
    try:
        doc_id = get_file_hash(pdf_path)
        print(f"✅ Document ID: {doc_id[:16]}... (truncated)")
    except Exception as e:
        print(f"❌ Error generating ID: {e}")
        return
    
    print(f"\n{'='*60}")
    print("✅ All tests passed successfully!")
    print(f"{'='*60}\n")
    
    # Summary
    print("Summary:")
    print(f"  - PDF: {Path(pdf_path).name}")
    print(f"  - Pages: {len(pages_text)}")
    print(f"  - Chunks: {len(chunks)}")
    print(f"  - Document ID: {doc_id[:16]}...")
    print(f"  - Ready to add to MCP server! ✨\n")


def test_semantic_chunking():
    """Test semantic chunking with example text."""
    print(f"\n{'='*60}")
    print("Testing Semantic Chunking Algorithm")
    print(f"{'='*60}\n")
    
    sample_text = """
    Natural language processing is a field of artificial intelligence. 
    It focuses on the interaction between computers and human language.
    Machine learning algorithms are commonly used in NLP tasks.
    These algorithms can learn patterns from large text datasets.
    Deep learning has revolutionized NLP in recent years.
    Transformer models like BERT and GPT have achieved state-of-the-art results.
    They use attention mechanisms to understand context in text.
    """
    
    print("Sample text:")
    print(sample_text.strip())
    print()
    
    # Test different chunk sizes
    for chunk_size in [2, 3, 4]:
        print(f"\nChunk size: {chunk_size} sentences, overlap: 1")
        print("-" * 40)
        chunks = semantic_chunking(sample_text, chunk_size=chunk_size, overlap=1)
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}: {chunk[:80]}...")
        print(f"  Total chunks: {len(chunks)}")


def main():
    """Main test function."""
    print("\n🔍 PDF RAG MCP Server - Test Suite\n")
    
    # Test 1: Semantic chunking algorithm
    test_semantic_chunking()
    
    # Test 2: PDF processing (if PDF provided)
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if Path(pdf_path).exists():
            test_pdf_processing(pdf_path)
        else:
            print(f"\n❌ Error: PDF file not found: {pdf_path}")
    else:
        print("\n" + "="*60)
        print("To test with a real PDF, run:")
        print(f"  python {Path(__file__).name} /path/to/your/document.pdf")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
