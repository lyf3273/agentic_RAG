"""BM25 + RRF 混合检索单元测试，不依赖向量库。"""

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import BaseRetriever

from hybrid_retriever import HybridRRFRetriever, JiebaBM25Retriever, rrf_fuse, tokenize_chinese


class _FixedRetriever(BaseRetriever):
    def __init__(self, hits):
        self._hits = hits
        super().__init__()

    def _retrieve(self, query_bundle):
        return self._hits


def test_tokenize_keeps_security_terms():
    tokens = tokenize_chinese("fastjson 反序列化漏洞如何利用 JNDI 注入")
    assert "fastjson" in tokens
    assert "jndi" in tokens
    assert "反序列化" in tokens
    assert "如何" not in tokens


def test_bm25_ranks_keyword_doc_first():
    nodes = [
        TextNode(id_="sql", text="SQL 注入的防御方法是使用预编译语句和参数绑定。"),
        TextNode(id_="fastjson", text="fastjson 反序列化漏洞利用常见路径是 JNDI 注入。"),
        TextNode(id_="ssrf", text="SSRF 可以探测内网服务并打云元数据。"),
    ]
    retriever = JiebaBM25Retriever(nodes, similarity_top_k=2)
    hits = retriever.retrieve("fastjson JNDI 反序列化")
    assert hits, "BM25 应返回结果"
    assert hits[0].node.node_id == "fastjson"


def test_rrf_promotes_overlap():
    a = TextNode(id_="a", text="a")
    b = TextNode(id_="b", text="b")
    c = TextNode(id_="c", text="c")
    vector_hits = [
        NodeWithScore(node=a, score=0.9),
        NodeWithScore(node=b, score=0.8),
        NodeWithScore(node=c, score=0.1),
    ]
    bm25_hits = [
        NodeWithScore(node=c, score=12.0),
        NodeWithScore(node=a, score=3.0),
    ]
    fused = rrf_fuse([vector_hits, bm25_hits], top_n=3, rrf_k=60)
    ids = [hit.node.node_id for hit in fused]
    assert ids[0] == "a"
    assert "c" in ids
    assert fused[0].score > fused[1].score


def test_hybrid_retriever_exposes_source_results():
    a = TextNode(id_="a", text="a")
    b = TextNode(id_="b", text="b")
    hybrid = HybridRRFRetriever(
        [
            ("vector", _FixedRetriever([NodeWithScore(node=a, score=0.9)])),
            ("bm25", _FixedRetriever([NodeWithScore(node=b, score=4.0), NodeWithScore(node=a, score=1.0)])),
        ],
        similarity_top_k=2,
        rrf_k=60,
    )
    hits = hybrid.retrieve("anything")
    assert [item.node.node_id for item in hits][0] == "a"
    names = [name for name, _ in hybrid.last_source_results]
    assert names == ["vector", "bm25"]
