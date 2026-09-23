"""Markdown 父子切分单测。"""

from llama_index.core import Document

from markdown_chunker import MarkdownParentChildSplitter, expand_retrieved_node


SAMPLE = """# AI Agent 入门

开篇说明 Agent 公式。

## 上下文工程

这一段是父节导语，解释上下文为什么重要。

### KV Cache

KV Cache 复用历史计算结果，能降低推理延迟。前缀稳定时命中率最高。

### 提示注入

提示注入是把恶意指令塞进上下文，让模型偏离系统提示。

## 工具

工具是 Agent 的手脚。

### MCP

MCP 用统一协议暴露工具。
"""


def test_heading_sections_keep_titles():
    splitter = MarkdownParentChildSplitter(child_max_chars=2000)
    nodes = splitter.get_nodes_from_documents([Document(text=SAMPLE, metadata={"file_name": "ch1.md"})])
    titles = [" ".join(n.get_content()[:40].split()) for n in nodes]
    assert any("KV Cache" in t for t in titles)
    assert any("MCP" in t for t in titles)


def test_long_section_falls_back_and_shares_parent():
    long_body = "这是一句用来撑长度的说明。" * 80
    text = f"# 章\n\n## 大节\n\n### 超长小节\n\n{long_body}"
    splitter = MarkdownParentChildSplitter(child_max_chars=400, parent_max_chars=2000)
    nodes = splitter.get_nodes_from_documents([Document(text=text, metadata={"file_name": "long.md"})])
    assert len(nodes) >= 2
    parent_ids = {n.metadata["parent_id"] for n in nodes if "超长" in n.get_content() or n.metadata.get("chunk_index") is not None}
    assert len(parent_ids) == 1


def test_expand_prefers_parent():
    splitter = MarkdownParentChildSplitter(child_max_chars=2000)
    nodes = splitter.get_nodes_from_documents([Document(text=SAMPLE, metadata={"file_name": "ch1.md"})])
    kv = next(n for n in nodes if "KV Cache" in n.get_content())
    expanded, path = expand_retrieved_node(kv)
    assert "提示注入" in expanded
    assert "KV Cache" in path
