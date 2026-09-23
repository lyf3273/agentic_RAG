"""L2-normalize embeddings so FAISS IndexFlatL2 ranks by cosine similarity.

Settings.embed_model only accepts llama_index BaseEmbedding. A plain wrapper
fails that isinstance check. Subclass BaseEmbedding and delegate the private
embedding methods, which is where every public API eventually lands.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import ConfigDict, Field


def l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec)) or 1.0
    return [float(x) / norm for x in vec]


class L2NormalizeEmbedding(BaseEmbedding):
    """BaseEmbedding that L2-normalizes every vector from an inner model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    inner: Any = Field(exclude=True)

    def __init__(self, inner: BaseEmbedding, **kwargs: Any) -> None:
        super().__init__(
            inner=inner,
            model_name=getattr(inner, "model_name", "unknown"),
            embed_batch_size=min(int(getattr(inner, "embed_batch_size", 10) or 10), 10),
            **kwargs,
        )

    def _get_query_embedding(self, query: str):
        return l2_normalize(self.inner._get_query_embedding(query))

    async def _aget_query_embedding(self, query: str):
        return l2_normalize(await self.inner._aget_query_embedding(query))

    def _get_text_embedding(self, text: str):
        return l2_normalize(self.inner._get_text_embedding(text))

    async def _aget_text_embedding(self, text: str):
        return l2_normalize(await self.inner._aget_text_embedding(text))

    def _get_text_embeddings(self, texts: list[str]):
        return [l2_normalize(v) for v in self.inner._get_text_embeddings(texts)]

    async def _aget_text_embeddings(self, texts: list[str]):
        vecs = await self.inner._aget_text_embeddings(texts)
        return [l2_normalize(v) for v in vecs]


def install_l2_norm(embed_model: Any) -> Any:
    """Return a BaseEmbedding wrapper. Safe to call twice."""
    if embed_model is None or isinstance(embed_model, L2NormalizeEmbedding):
        return embed_model
    if not isinstance(embed_model, BaseEmbedding):
        raise TypeError(
            f"embed model must be llama_index BaseEmbedding, got {type(embed_model).__name__}"
        )
    return L2NormalizeEmbedding(embed_model)
