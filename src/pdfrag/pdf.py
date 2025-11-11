# ABOUTME: Extracts text from PDF files using PyMuPDF with OCR fallback.
# ABOUTME: Handles both standard text extraction and scanned document OCR.

"""PDF text extraction with OCR support."""

from typing import List, Dict, Any
import fitz  # PyMuPDF

# Minimum characters to consider a page has text (not scanned)
MIN_TEXT_THRESHOLD = 50


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF with page numbers, using OCR for scanned pages.

    Tries standard text extraction first. If a page has minimal text (likely scanned),
    falls back to OCR using PyMuPDF's built-in Tesseract integration.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with 'page', 'text', and 'ocr_used' keys

    Example:
        >>> pages = extract_text_from_pdf("/path/to/doc.pdf")
        >>> print(f"Extracted {len(pages)} pages")
        >>> print(f"Page 1: {pages[0]['text'][:100]}")
    """
    pages_text = []

    # Open PDF with PyMuPDF
    doc = fitz.open(pdf_path)

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            ocr_used = False

            # Try standard text extraction first
            text = page.get_text()

            # If minimal text found, likely a scanned page - use OCR
            if len(text.strip()) < MIN_TEXT_THRESHOLD:
                try:
                    # Use PyMuPDF's OCR capability (requires tesseract)
                    text_page = page.get_textpage_ocr()
                    text = page.get_text(textpage=text_page)
                    ocr_used = True
                except Exception:
                    # OCR failed (tesseract not installed or other error)
                    # Fall back to whatever text we got, even if minimal
                    pass

            # Only add pages with text content
            if text.strip():
                pages_text.append({
                    'page': page_num + 1,  # 1-indexed for user display
                    'text': text,
                    'ocr_used': ocr_used
                })

    finally:
        doc.close()

    return pages_text
