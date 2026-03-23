# ABOUTME: Tests for PDFDatabase operations, including correctness with large collections.
# ABOUTME: Covers document_exists, list_documents, and search_keywords with >32k chunks (SQLite variable limit).

from pdfrag.database import PDFDatabase

# Number of chunks needed to exceed SQLite's variable limit in collection.get()
LARGE_CHUNK_COUNT = 33000


def _make_db(tmp_path):
    """Create a PDFDatabase backed by tmp_path."""
    return PDFDatabase(str(tmp_path))


def _populate_large(db, doc_id="doc_large", filename="large.pdf", n=LARGE_CHUNK_COUNT):
    """Insert n chunks with 1-D dummy embeddings into db under doc_id."""
    chunks = [
        {"text": f"chunk {i}", "page": i % 10, "chunk_index": i}
        for i in range(n)
    ]
    embeddings = [[0.1] for _ in range(n)]
    db.add_document(doc_id, filename, chunks, embeddings)
    return chunks


def test_document_exists_returns_false_when_empty(tmp_path):
    db = _make_db(tmp_path)
    assert db.document_exists("nonexistent") is False


def test_document_exists_returns_true_after_add(tmp_path):
    db = _make_db(tmp_path)
    chunks = [{"text": "hello", "page": 1, "chunk_index": 0}]
    embeddings = [[0.1]]
    db.add_document("doc1", "test.pdf", chunks, embeddings)
    assert db.document_exists("doc1") is True


def test_document_exists_large_collection(tmp_path):
    """document_exists must not fail when a document has >32k chunks."""
    db = _make_db(tmp_path)
    _populate_large(db)
    # Should return True without raising "too many SQL variables"
    assert db.document_exists("doc_large") is True


def test_list_documents_large_collection(tmp_path):
    """list_documents must not fail when a document has >32k chunks."""
    db = _make_db(tmp_path)
    _populate_large(db)
    docs = db.list_documents()
    assert len(docs) == 1
    assert docs[0]["document_id"] == "doc_large"
    assert docs[0]["filename"] == "large.pdf"
    assert docs[0]["chunk_count"] == LARGE_CHUNK_COUNT


def test_list_documents_multiple_large_docs(tmp_path):
    """list_documents correctly counts chunks across multiple large documents."""
    db = _make_db(tmp_path)
    _populate_large(db, doc_id="doc_a", filename="a.pdf", n=LARGE_CHUNK_COUNT)
    _populate_large(db, doc_id="doc_b", filename="b.pdf", n=100)
    docs = db.list_documents()
    assert len(docs) == 2
    by_id = {d["document_id"]: d for d in docs}
    assert by_id["doc_a"]["chunk_count"] == LARGE_CHUNK_COUNT
    assert by_id["doc_b"]["chunk_count"] == 100


def test_search_keywords_large_collection(tmp_path):
    """search_keywords must not fail when the collection has >32k chunks."""
    db = _make_db(tmp_path)
    needle_indices = {0, 1000, 32000}
    chunks = []
    for i in range(LARGE_CHUNK_COUNT):
        text = "needle unique term" if i in needle_indices else f"chunk {i}"
        chunks.append({"text": text, "page": i % 10, "chunk_index": i})
    embeddings = [[0.1] for _ in range(LARGE_CHUNK_COUNT)]
    db.add_document("doc_large", "large.pdf", chunks, embeddings)

    results = db.search_keywords(["needle"], top_k=5)
    assert len(results) == len(needle_indices)
    assert all("needle" in r["text"] for r in results)
