"""结构化 Markdown 切分：标题语义边界 + 超长句切兜底 + 父子块。

检索用子块（更准），回答时回填父段（更完整）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from llama_index.core.bridge.pydantic import Field
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.file.markdown import MarkdownNodeParser
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, Document, MetadataMode, TextNode

# 子块：检索/精排单位。超过则在该标题段内按句切。
CHILD_MAX_CHARS = 1800
# 父块：回填给 LLM 的上下文上限，避免整章灌进去。
PARENT_MAX_CHARS = 3500
# 与历史索引接近的句切参数，仅作超长段兜底。
FALLBACK_CHUNK_SIZE = 512
FALLBACK_CHUNK_OVERLAP = 80


def _heading_title(text: str) -> str:
    first = (text or "").split("\n", 1)[0].strip()
    return re.sub(r"^#{1,6}\s+", "", first).strip()


def _normalize_path(header_path: str, title: str) -> list[str]:
    parts = [p for p in (header_path or "").strip("/").split("/") if p]
    if title and (not parts or parts[-1] != title):
        parts.append(title)
    return parts


def _parent_id(file_name: str, parts: list[str]) -> str:
    key = "/".join(parts[:2]) if parts else file_name
    return f"{file_name}::{key}"


class MarkdownParentChildSplitter(NodeParser):
    """先按 Markdown 标题切，超长段再用 SentenceSplitter 兜底，并挂上父段文本。"""

    child_max_chars: int = Field(default=CHILD_MAX_CHARS)
    parent_max_chars: int = Field(default=PARENT_MAX_CHARS)
    fallback_chunk_size: int = Field(default=FALLBACK_CHUNK_SIZE)
    fallback_chunk_overlap: int = Field(default=FALLBACK_CHUNK_OVERLAP)

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs,
    ) -> list[BaseNode]:
        markdown = MarkdownNodeParser.from_defaults(
            include_metadata=True,
            include_prev_next_rel=True,
        )
        fallback = SentenceSplitter(
            chunk_size=self.fallback_chunk_size,
            chunk_overlap=self.fallback_chunk_overlap,
        )
        sections: list[TextNode] = []
        for node in nodes:
            inherited = {
                key: node.metadata[key]
                for key in (
                    "file_name",
                    "source_path",
                    "topic",
                    "doc_hash",
                    "doc_mtime",
                    "vector_dim",
                    "chunker",
                )
                if node.metadata.get(key) not in (None, "")
            }
            for section in markdown.get_nodes_from_node(node):
                section.metadata.update(inherited)
                sections.append(section)

        grouped: dict[str, list[str]] = {}
        prepared: list[tuple[TextNode, str, list[str]]] = []
        for section in sections:
            text = section.get_content(metadata_mode=MetadataMode.NONE).strip()
            if not text:
                continue
            if not text.startswith("#") and len(text) < 80:
                continue
            file_name = section.metadata.get("file_name", "")
            title = _heading_title(text)
            parts = _normalize_path(section.metadata.get("header_path", ""), title)
            pid = _parent_id(file_name, parts)
            grouped.setdefault(pid, []).append(text)
            prepared.append((section, pid, parts))

        parent_texts = {pid: "\n\n".join(chunks) for pid, chunks in grouped.items()}
        children: list[BaseNode] = []
        for section, pid, parts in prepared:
            text = section.get_content(metadata_mode=MetadataMode.NONE).strip()
            header_path = "/".join(parts)
            parent_text = self._window_parent(parent_texts[pid], text)
            extra = {
                "header_path": header_path,
                "parent_id": pid,
                "parent_text": parent_text,
            }
            if len(text) <= self.child_max_chars:
                section.metadata.update(extra)
                children.append(section)
                continue
            doc = Document(text=text, metadata=dict(section.metadata))
            for idx, sub in enumerate(fallback.get_nodes_from_documents([doc])):
                sub.metadata.update(extra)
                sub.metadata["chunk_index"] = idx
                children.append(sub)
        return children

    def _window_parent(self, combined: str, child_text: str) -> str:
        if len(combined) <= self.parent_max_chars:
            return combined
        idx = combined.find(child_text)
        if idx < 0:
            return combined[: self.parent_max_chars]
        start = max(0, idx - self.parent_max_chars // 4)
        return combined[start : start + self.parent_max_chars]


_PARENT_CACHE: dict[str, str] | None = None
_PARENT_MTIME: float | None = None


def _parent_from_sidecar(node_id: str) -> str:
    global _PARENT_CACHE, _PARENT_MTIME
    path = Path(__file__).resolve().parent / ".cache" / "parent_texts.json"
    if not path.is_file():
        return ""
    mtime = path.stat().st_mtime
    if _PARENT_CACHE is None or _PARENT_MTIME != mtime:
        try:
            _PARENT_CACHE = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            _PARENT_CACHE = {}
        _PARENT_MTIME = mtime
    return (_PARENT_CACHE.get(node_id) or "").strip()


def expand_retrieved_node(node: BaseNode) -> tuple[str, str]:
    """返回 (展示文本, 标题路径)。多个子块命中同一父段时应在外层按 parent_id 去重。"""
    content = node.get_content(metadata_mode=MetadataMode.NONE).strip()
    parent = (node.metadata.get("parent_text") or "").strip()
    if not parent:
        parent = _parent_from_sidecar(getattr(node, "node_id", "") or "")
    header_path = (node.metadata.get("header_path") or "").strip("/")
    if parent and len(parent) > len(content) * 1.15:
        return parent, header_path
    return content, header_path
