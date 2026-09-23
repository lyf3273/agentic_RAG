"""BM25 + 向量检索 + RRF 混合检索。"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sequence

import bm25s
import jieba
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
)

_JIEBA_READY = False
_JIEBA_LOCK = threading.Lock()
_TOKEN_KEEP = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]")
_CN_STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "吗", "呢", "啊", "把", "被", "与", "及", "或", "等",
    "如何", "什么", "怎么", "怎样", "请问", "一下", "这个", "那个", "可以",
    "以及", "如果", "因为", "所以", "然后", "但是", "而且", "对于", "关于",
}

_DOMAIN_WORDS = (
    "fastjson", "log4j", "Log4Shell", "shiro", "JNDI", "XXE", "SSRF",
    "nacos", "xxl-job", "weblogic", "websphere", "memshell", "反序列化",
    "KV Cache", "Function Calling", "MCP", "ReAct", "Harness",
    "上下文工程", "提示注入", "多Agent",
    "HyDE", "RRF", "Cross-Encoder", "FAISS", "ChromaDB",
)


def _ensure_jieba() -> None:
    global _JIEBA_READY
    if _JIEBA_READY:
        return
    with _JIEBA_LOCK:
        if _JIEBA_READY:
            return
        jieba.setLogLevel(logging.INFO)
        jieba.initialize()
        for word in _DOMAIN_WORDS:
            jieba.add_word(word)
        _JIEBA_READY = True


def tokenize_chinese(text: str) -> list[str]:
    """jieba 搜索分词，供 BM25 使用。"""
    _ensure_jieba()
    with _JIEBA_LOCK:
        raw_tokens = jieba.lcut_for_search(text or "")
    tokens: list[str] = []
    for raw in raw_tokens:
        token = raw.strip().lower()
        if not token or token in _CN_STOPWORDS:
            continue
        if not _TOKEN_KEEP.search(token):
            continue
        if len(token) == 1 and "\u4e00" <= token <= "\u9fff":
            continue
        tokens.append(token)
    return tokens


def _node_key(node: BaseNode) -> str:
    node_id = getattr(node, "node_id", None)
    if node_id:
        return f"id:{node_id}"
    text = (node.get_content(metadata_mode="none") or "").strip()
    return "md5:" + hashlib.md5(text.encode("utf-8")).hexdigest()


def _nodes_from_chroma_collection(collection) -> list[BaseNode]:
    result = collection.get(include=["documents", "metadatas"])
    ids = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    nodes: list[BaseNode] = []
    for i, node_id in enumerate(ids):
        text = docs[i] if i < len(docs) else ""
        metadata = metas[i] if i < len(metas) else {}
        metadata = metadata or {}
        node: BaseNode
        try:
            node = metadata_dict_to_node(metadata, text=text)
        except Exception:
            try:
                md, node_info, relationships = legacy_metadata_dict_to_node(metadata)
                node = TextNode(
                    text=text or "",
                    id_=node_id,
                    metadata=md,
                    start_char_idx=node_info.get("start"),
                    end_char_idx=node_info.get("end"),
                    relationships=relationships,
                )
            except Exception:
                node = TextNode(text=text or "", id_=node_id, metadata=metadata)
        nodes.append(node)
    return nodes


def load_all_nodes(index) -> list[BaseNode]:
    """从 Chroma / FAISS / docstore 抽出全部文本节点，供 BM25 建索引。"""
    nodes: list[BaseNode] = []
    vector_store = getattr(index, "vector_store", None)
    collection = getattr(vector_store, "_collection", None) if vector_store is not None else None
    if collection is not None:
        try:
            nodes = _nodes_from_chroma_collection(collection)
        except Exception as exc:
            print(f"[!] 从 Chroma 提取节点失败: {exc}")
            nodes = []
    if not nodes and vector_store is not None and type(vector_store).__name__ != "FaissVectorStore":
        getter = getattr(vector_store, "get_nodes", None)
        if callable(getter):
            try:
                nodes = list(getter(None) or [])
            except NotImplementedError:
                nodes = []
            except Exception as exc:
                print(f"[!] vector_store.get_nodes 失败: {exc}")
                nodes = []
    if not nodes:
        docs = getattr(getattr(index, "docstore", None), "docs", {}) or {}
        nodes = list(docs.values())
    nodes = [
        node
        for node in nodes
        if hasattr(node, "get_content")
        and (node.get_content(metadata_mode="none") or "").strip()
    ]
    if not nodes:
        raise RuntimeError("无法从索引提取文本节点，无法构建 BM25")
    return nodes


class JiebaBM25Retriever(BaseRetriever):
    """中文 BM25 检索：jieba 分词 + bm25s。"""

    def __init__(self, nodes: Sequence[BaseNode], similarity_top_k: int = 10) -> None:
        self._nodes = list(nodes)
        self.similarity_top_k = similarity_top_k
        corpus_tokens = [
            tokenize_chinese(node.get_content(metadata_mode="none"))
            for node in self._nodes
        ]
        # 空文档用占位符，避免 bm25s 跳过导致下标错位
        corpus_tokens = [tokens or ["_empty"] for tokens in corpus_tokens]
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens, show_progress=False)
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_tokens = tokenize_chinese(query_bundle.query_str)
        if not query_tokens or not self._nodes:
            return []
        top_k = min(self.similarity_top_k, len(self._nodes))
        try:
            indexes, scores = self._bm25.retrieve(
                [query_tokens],
                k=top_k,
                show_progress=False,
            )
        except Exception:
            return []
        hits: list[NodeWithScore] = []
        for idx, score in zip(indexes[0], scores[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(self._nodes):
                continue
            hits.append(NodeWithScore(node=self._nodes[idx], score=float(score)))
        return hits


def rrf_fuse(
    result_lists: Iterable[Sequence[NodeWithScore]],
    *,
    top_n: int,
    rrf_k: int = 60,
) -> list[NodeWithScore]:
    """标准 Reciprocal Rank Fusion：score = Σ 1/(k + rank)。"""
    fused: dict[str, float] = defaultdict(float)
    node_map: dict[str, BaseNode] = {}
    for hits in result_lists:
        ranked = sorted(hits, key=lambda item: item.score or 0.0, reverse=True)
        for rank, hit in enumerate(ranked, start=1):
            key = _node_key(hit.node)
            fused[key] += 1.0 / (rrf_k + rank)
            if key not in node_map:
                node_map[key] = hit.node
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [NodeWithScore(node=node_map[key], score=score) for key, score in ordered]


class HybridRRFRetriever(BaseRetriever):
    """多路检索后做 RRF 融合。BM25 与向量路并行；分路结果放 thread-local，避免并发串台。"""

    def __init__(
        self,
        retrievers: Sequence[tuple[str, BaseRetriever]],
        similarity_top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("hybrid retriever 至少需要一路检索器")
        self._retrievers = list(retrievers)
        self.similarity_top_k = similarity_top_k
        self.rrf_k = rrf_k
        self._tls = threading.local()
        workers = max(1, min(4, len(self._retrievers)))
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hybrid-rrf")
        super().__init__()

    @property
    def last_source_results(self) -> list[tuple[str, list[NodeWithScore]]]:
        return getattr(self._tls, "value", [])

    @last_source_results.setter
    def last_source_results(self, value: list[tuple[str, list[NodeWithScore]]]) -> None:
        self._tls.value = value

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if len(self._retrievers) == 1:
            name, retriever = self._retrievers[0]
            source_results = [(name, retriever.retrieve(query_bundle))]
        else:
            futures = [
                (name, self._pool.submit(retriever.retrieve, query_bundle))
                for name, retriever in self._retrievers
            ]
            source_results = [(name, fut.result()) for name, fut in futures]
        self.last_source_results = source_results
        return rrf_fuse(
            [hits for _, hits in source_results],
            top_n=self.similarity_top_k,
            rrf_k=self.rrf_k,
        )


def build_hybrid_retriever(
    index,
    *,
    vector_top_k: int = 10,
    bm25_top_k: int = 10,
    fusion_top_k: int = 5,
    rrf_k: int = 60,
) -> tuple[HybridRRFRetriever, BaseRetriever, JiebaBM25Retriever]:
    """构建 BM25 + 向量 + RRF 混合检索器。"""
    nodes = load_all_nodes(index)
    print(f"[*] BM25 语料节点数: {len(nodes)}")
    vector_retriever = index.as_retriever(similarity_top_k=vector_top_k)
    bm25_retriever = JiebaBM25Retriever(nodes, similarity_top_k=bm25_top_k)
    hybrid = HybridRRFRetriever(
        [("vector", vector_retriever), ("bm25", bm25_retriever)],
        similarity_top_k=fusion_top_k,
        rrf_k=rrf_k,
    )
    return hybrid, vector_retriever, bm25_retriever
