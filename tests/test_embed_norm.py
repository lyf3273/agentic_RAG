from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from embed_norm import L2NormalizeEmbedding, install_l2_norm, l2_normalize


class _FakeEmbed(BaseEmbedding):
    def _get_text_embedding(self, text: str):
        return [3.0, 4.0]

    def _get_query_embedding(self, query: str):
        return [6.0, 8.0]

    async def _aget_query_embedding(self, query: str):
        return [6.0, 8.0]

    def _get_text_embeddings(self, texts):
        return [[3.0, 4.0] for _ in texts]


def test_l2_normalize_unit_length():
    vec = l2_normalize([3.0, 4.0])
    assert abs(vec[0] - 0.6) < 1e-9
    assert abs(vec[1] - 0.8) < 1e-9


def test_install_is_idempotent_and_unit():
    wrapped = install_l2_norm(install_l2_norm(_FakeEmbed(model_name="fake")))
    assert isinstance(wrapped, BaseEmbedding)
    q = wrapped.get_query_embedding("x")
    assert abs(sum(x * x for x in q) - 1.0) < 1e-9
    batch = wrapped.get_text_embedding_batch(["a", "b"])
    assert len(batch) == 2
    assert abs(sum(x * x for x in batch[0]) - 1.0) < 1e-9


def test_openai_embedding_passes_settings_type_check():
    inner = OpenAIEmbedding(
        model_name="text-embedding-v3",
        api_key="sk-test",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions=1024,
    )
    wrapped = install_l2_norm(inner)
    assert isinstance(wrapped, L2NormalizeEmbedding)
    assert isinstance(wrapped, BaseEmbedding)
