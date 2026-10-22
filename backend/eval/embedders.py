"""Embedders for the evaluation.

HashingEmbeddings is a TEST DOUBLE. It turns words into a fixed-size vector by hashing them, so it has no idea that
"threat" and "risk" mean the same thing. It exists so the harness, the ranking code and the baseline can run in CI with
no model download and no network. Numbers from it prove the harness works; they say nothing about retrieval quality.
"""
import hashlib
import math
import re
from typing import List

from langchain_core.embeddings import Embeddings

# Function words carry no topic; dropping them keeps the hashed vectors from being dominated by "the" and "of".
_STOPWORDS = frozenset(
    "a an and are as at be by does do for from has have how in is it its of on or that the their there this to "
    "was were what when which who whom whose will with".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase words and numbers, stopwords dropped, a trailing plural s trimmed so "risks" meets "risk"."""
    tokens = []
    for token in _TOKEN.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


class HashingEmbeddings(Embeddings):
    """Signed feature hashing into `dim` buckets, L2-normalised so a dot product is a cosine.

    It SUBCLASSES langchain_core's Embeddings on purpose: QdrantVectorStore type-checks its embedding argument and
    rejects a structurally identical duck type with a TypeError (the same trap embeddings.py documents).

    blake2b, not Python's hash(): hash() is salted per process, which would make every run different.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vector(self, text: str) -> List[float]:
        counts = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
        vec = [0.0] * self.dim
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vec[value % self.dim] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:  # a text of only stopwords: a fixed unit vector keeps cosine defined
            vec[0] = 1.0
            return vec
        return [x / norm for x in vec]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def real_embeddings():
    """The repo's own default embedding model (torch based when torch is installed, else the fastembed ONNX adapter).

    Returns (embeddings, backend_label). The label names the class that actually ran, so a table never claims a
    backend it did not use.
    """
    from app.services.embeddings import embedding_service

    emb = embedding_service.get_embeddings()
    return emb, type(emb).__name__
