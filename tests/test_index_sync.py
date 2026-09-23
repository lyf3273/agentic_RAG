"""chunk_id / 文档指纹 / 改文档先删后建。"""

from pathlib import Path

from llama_index.core import Document

from index_sync import (
    CHUNKER_VERSION,
    delete_doc_chunks,
    file_fingerprint,
    make_chunk_id,
    make_node_id,
    scan_indexed_docs,
    stamp_nodes,
)
from markdown_chunker import MarkdownParentChildSplitter


def test_chunk_id_is_filename_plus_seq():
    assert make_chunk_id("01-AI-Agent入门.md", 1) == "01-AI-Agent入门.md::0001"
    assert make_node_id("ai-agent-book/01-AI-Agent入门.md", 3) == "ai-agent-book/01-AI-Agent入门.md::0003"


def test_stamp_nodes_hash_mtime_and_ids():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    path = data_dir / "ai-agent-book" / "00-引言.md"
    if not path.is_file():
        return
    doc_hash, doc_mtime = file_fingerprint(path)
    text = path.read_text(encoding="utf-8")
    doc = Document(text=text, metadata={"file_name": path.name, "source_path": path.relative_to(data_dir).as_posix()})
    nodes = MarkdownParentChildSplitter().get_nodes_from_documents([doc])
    stamped = stamp_nodes(nodes, path=path, data_dir=data_dir, doc_hash=doc_hash, doc_mtime=doc_mtime)
    assert stamped
    first = stamped[0]
    assert first.metadata["chunk_id"] == make_chunk_id(path.name, 1)
    assert first.metadata["doc_hash"] == doc_hash
    assert first.metadata["chunker"] == CHUNKER_VERSION
    assert first.node_id.endswith("::0001")


class _FakeCollection:
    def __init__(self, rows: list[tuple[str, dict]]):
        self.rows = list(rows)

    def count(self):
        return len(self.rows)

    def get(self, where=None, include=None, limit=None, offset=None):
        rows = self.rows
        if where:
            key, val = next(iter(where.items()))
            rows = [(i, m) for i, m in rows if (m or {}).get(key) == val]
        if offset:
            rows = rows[offset:]
        if limit:
            rows = rows[:limit]
        return {"ids": [i for i, _ in rows], "metadatas": [m for _, m in rows]}

    def delete(self, ids=None, where=None):
        drop = set(ids or [])
        self.rows = [(i, m) for i, m in self.rows if i not in drop]


def test_scan_and_delete_by_source_path():
    col = _FakeCollection(
        [
            ("a.md::0001", {"source_path": "a.md", "doc_hash": "aaa", "chunker": CHUNKER_VERSION}),
            ("a.md::0002", {"source_path": "a.md", "doc_hash": "aaa", "chunker": CHUNKER_VERSION}),
            ("b.md::0001", {"source_path": "b.md", "doc_hash": "bbb", "chunker": CHUNKER_VERSION}),
        ]
    )
    scanned = scan_indexed_docs(col)
    assert scanned["a.md"]["doc_hash"] == "aaa"
    assert len(scanned["a.md"]["ids"]) == 2
    n = delete_doc_chunks(col, "a.md", extra_ids=scanned["a.md"]["ids"])
    assert n == 2
    left = scan_indexed_docs(col)
    assert "a.md" not in left
    assert "b.md" in left
