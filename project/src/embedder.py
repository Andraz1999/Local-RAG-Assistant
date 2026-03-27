"""
embedder.py
-----------
All text encoders for the RAG pipeline, separated by type.

Dense encoders (return np.ndarray of shape (n, dim)):
    OllamaEmbedder       — via local Ollama server
    HuggingFaceEmbedder  — via local HuggingFace model

Sparse neural encoder (returns list of sparse dicts {token_id: weight}):
    SpladeEncoder        — via local HuggingFace model

Sparse lexical encoder (statistical, corpus-level, no vectors):
    BM25Encoder          — builds a corpus index, scores all docs per query

Factory:
    get_encoders(config) → (DenseEmbedder, SpladeEncoder, BM25Encoder)

Recommended models:
    Dense  (Ollama):      nomic-embed-text
    Dense  (HuggingFace): BAAI/bge-large-en-v1.5
    SPLADE:               naver/splade-cocondenser-ensembledistil
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np # type: ignore

from src.config_loader import get_embedding_model, get_splade_model


# ===========================================================================
# DENSE
# ===========================================================================

class BaseDenseEmbedder(abc.ABC):
    """Encodes texts into fixed-size float vectors."""

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Args:
            texts: Non-empty list of strings.
        Returns:
            Float32 array of shape (len(texts), embedding_dim).
        """

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class OllamaEmbedder(BaseDenseEmbedder):
    """
    Dense embedder using the Ollama REST API.
    Requirements: pip install httpx
    Setup:        ollama pull nomic-embed-text
    """

    def __init__(self, config: dict[str, Any]) -> None:
        import httpx # type: ignore

        cfg = config["embedding"]
        _, self._model      = get_embedding_model(config)
        self._base_url   = cfg.get("ollama_base_url", "http://localhost:11434")
        self._batch_size = cfg.get("batch_size", 32)
        self._client     = httpx.Client(base_url=self._base_url, timeout=120)

    def embed(self, texts: list[str]) -> np.ndarray:
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch    = texts[i : i + self._batch_size]
            response = self._client.post(
                "/api/embed",
                json={"model": self._model, "input": batch},
            )
            response.raise_for_status()
            all_vectors.extend(response.json()["embeddings"])
        return np.array(all_vectors, dtype=np.float32)


class HuggingFaceEmbedder(BaseDenseEmbedder):
    """
    Dense embedder using a local sentence-transformers model.
    Requirements: pip install sentence-transformers
    """

    def __init__(self, config: dict[str, Any]) -> None:
        from sentence_transformers import SentenceTransformer # type: ignore

        cfg        = config["embedding"]
        _, model_name = get_embedding_model(config)
        self._batch_size = cfg.get("batch_size", 32)

        print(f"Loading dense model: {model_name} ...")
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,      # cosine similarity ≡ dot product
            show_progress_bar=len(texts) > 64,
        )
        return np.array(vectors, dtype=np.float32)


# ===========================================================================
# SPLADE  (sparse neural)
# ===========================================================================

class SpladeEncoder:
    """
    Sparse neural encoder. Produces sparse vectors {token_id: weight}.
    Each text is encoded independently (like dense), but the output is a
    sparse dict rather than a float array. Similarity is a sparse dot product,
    handled here so vector_store.py stays format-agnostic.

    Requirements: pip install transformers torch
    """

    def __init__(self, config: dict[str, Any]) -> None:
        from transformers import AutoTokenizer, AutoModelForMaskedLM # type: ignore
        import torch # type: ignore

        cfg              = config["embedding"]
        model_name       = get_splade_model(config)
        self._batch_size = cfg.get("batch_size", 32)

        print(f"Loading SPLADE model: {model_name} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model     = AutoModelForMaskedLM.from_pretrained(model_name)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        print(f"  SPLADE running on: {self._device}")

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        """
        Encode texts into sparse vectors.

        Returns:
            List of dicts {token_id: weight}, one per text.
            Only non-zero weights are stored.
        """
        import torch # type: ignore

        results: list[dict[int, float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch  = texts[i : i + self._batch_size]
            tokens = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                output = self._model(**tokens)

            # Aggregate over sequence positions: log(1 + relu(logits)).max(seq)
            # logits: (batch, seq_len, vocab_size) → vecs: (batch, vocab_size)
            vecs = torch.log(1 + torch.relu(output.logits)).max(dim=1).values

            for vec in vecs:
                nonzero_ids = vec.nonzero(as_tuple=True)[0]
                results.append({
                    int(idx): float(vec[idx])
                    for idx in nonzero_ids
                    if float(vec[idx]) > 0
                })

        return results

    def encode_one(self, text: str) -> dict[int, float]:
        return self.encode([text])[0]

    @staticmethod
    def similarity(query_vec: dict[int, float], doc_vec: dict[int, float]) -> float:
        """
        Sparse dot product between query and document vectors.
        Only token IDs present in both vectors contribute to the score.
        Implemented as a static method so vector_store.py never needs to
        know the internal format of SPLADE vectors.
        """
        return sum(query_vec.get(k, 0.0) * v for k, v in doc_vec.items())


# ===========================================================================
# BM25  (sparse lexical — corpus-level, not per-document)
# ===========================================================================

class BM25Encoder:
    """
    Statistical sparse encoder based on term frequency and inverse document
    frequency. Unlike dense and SPLADE, BM25 cannot produce an independent
    vector per document — scores are relative to the entire corpus.

    Workflow:
        encoder = BM25Encoder()
        encoder.build_index(all_texts)          # call after any corpus change
        scores = encoder.get_scores("my query") # np.ndarray, one score per doc

    Requirements: pip install rank-bm25
    """

    def __init__(self) -> None:
        from rank_bm25 import BM25Okapi # type: ignore
        self._BM25Okapi = BM25Okapi
        self._index: Any = None

    def build_index(self, texts: list[str]) -> None:
        """
        Build (or rebuild) the BM25 index from the full corpus.
        Must be called from scratch whenever any document is added or removed.

        Args:
            texts: All document texts currently in the corpus.
        """
        tokenized     = [self._tokenize(t) for t in texts]
        self._index   = self._BM25Okapi(tokenized) if tokenized else None

    def get_scores(self, query: str) -> np.ndarray:
        """
        Score all documents against a query.

        Args:
            query: Raw query string (tokenized internally).
        Returns:
            Float array of shape (n_docs,), one BM25 score per document.
        """
        if self._index is None:
            return np.array([])
        return self._index.get_scores(self._tokenize(query))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return re.findall(r"\b\w+\b", text.lower())


# ===========================================================================
# Factory
# ===========================================================================

def get_encoders(
    config: dict[str, Any],
) -> tuple[BaseDenseEmbedder, SpladeEncoder, BM25Encoder]:
    """
    Instantiate and return all three encoders based on config.

    Args:
        config: Loaded project config.

    Returns:
        (dense_embedder, splade_encoder, bm25_encoder)
    """
    # Dense
    provider, _ = get_embedding_model(config)
    if provider == "ollama":
        dense = OllamaEmbedder(config)
    elif provider == "huggingface":
        dense = HuggingFaceEmbedder(config)
    else:
        raise ValueError(f"Unknown embedding provider: '{provider}'")

    splade = SpladeEncoder(config)
    bm25   = BM25Encoder()

    return dense, splade, bm25