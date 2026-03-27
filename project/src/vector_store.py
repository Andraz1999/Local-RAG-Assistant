"""
vector_store.py
---------------
Hybrid vector store maintaining three indexes in parallel:

    1. FAISS        — dense vectors, exact inner-product (cosine on L2-normalised)
    2. SPLADE       — sparse neural vectors, brute-force dot product
    3. BM25         — lexical index, persisted to disk

Indexing workflow (orchestrated by pipeline.py):
    1. remove_from_dense_and_splade(pdf_id)  — for deleted and changed PDFs
    2. add_to_dense_and_splade(chunks)        — for new and changed PDFs
    3. build_bm25()                           — once at the end of the pipeline run

Chunks already carry all required fields (pdf_id, source, chunk_index,
last_modified, metadata) so no external IDs or fingerprints are needed.

On-disk layout inside db_folder/:
    faiss.index      — FAISS IndexIDMap
    splade.pkl       — list[dict[int, float]], parallel to metadata
    bm25.pkl         — serialised BM25Okapi index object
    metadata.json    — list of chunk dicts
    registry.json    — {pdf_id: {source, last_modified}}
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss # type: ignore
import numpy as np # type: ignore

from .embedder import BM25Encoder, BaseDenseEmbedder, SpladeEncoder
from .query_rewriter import RewrittenQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows so inner product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _rrf_score(rank: int, constant: int) -> float:
    return 1.0 / (rank + constant)


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Maintains three parallel indexes (FAISS, SPLADE, BM25) over the same
    chunk corpus.

    FAISS and SPLADE are updated incrementally per pipeline run.
    BM25 is rebuilt once at the end of a pipeline run via build_bm25().

    Args:
        config:         Loaded project config.
        dense_embedder: OllamaEmbedder or HuggingFaceEmbedder.
        splade_encoder: SpladeEncoder instance.
        bm25_encoder:   BM25Encoder instance.
    """

    def __init__(
        self,
        config:         dict[str, Any],
        dense_embedder: BaseDenseEmbedder,
        splade_encoder: SpladeEncoder,
        bm25_encoder:   BM25Encoder,
    ) -> None:
        self._config = config
        self._dense  = dense_embedder
        self._splade = splade_encoder
        self._bm25   = bm25_encoder

        db_folder = Path(config["paths"]["db_folder"])
        db_folder.mkdir(parents=True, exist_ok=True)
        self._db_folder = db_folder

        self._faiss_path    = db_folder / "faiss.index"
        self._splade_path   = db_folder / "splade.pkl"
        self._bm25_path     = db_folder / "bm25.pkl"
        self._metadata_path = db_folder / "metadata.json"
        self._registry_path = db_folder / "registry.json"

        # Runtime state — all parallel to each other by position
        self._metadata:    list[dict[str, Any]]  = []
        self._splade_vecs: list[dict[int, float]] = []
        self._faiss_index: faiss.Index | None     = None

        self.registry: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all indexes from disk."""
        if self._metadata_path.exists():
            with open(self._metadata_path, "r") as f:
                self._metadata = json.load(f)
            print(f"Loaded {len(self._metadata)} chunks from metadata store.")
        else:
            self._metadata = []

        if self._faiss_path.exists():
            self._faiss_index = faiss.read_index(str(self._faiss_path))
            print(f"Loaded FAISS index ({self._faiss_index.ntotal} vectors).") # type: ignore
        else:
            self._faiss_index = None

        if self._splade_path.exists():
            with open(self._splade_path, "rb") as f:
                self._splade_vecs = pickle.load(f)
            print(f"Loaded SPLADE index ({len(self._splade_vecs)} vectors).")
        else:
            self._splade_vecs = []

        if self._bm25_path.exists():
            with open(self._bm25_path, "rb") as f:
                self._bm25._index = pickle.load(f)
            #print(f"Loaded BM25 index ({len(self._bm25._index)} vectors).")
        else:
            self._bm25._index = None

        if self._registry_path.exists():
            with open(self._registry_path, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def save(self) -> None:
        """Persist all three indexes, metadata, and registry to disk."""
        with open(self._metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(self._faiss_path))

        with open(self._splade_path, "wb") as f:
            pickle.dump(self._splade_vecs, f)

        if self._bm25._index is not None:
            with open(self._bm25_path, "wb") as f:
                pickle.dump(self._bm25._index, f)

        with open(self._registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

        print(f"Saved DB ({len(self._metadata)} chunks) to {self._db_folder}/")

    def reset(self) -> None:
        """
        Wipe all indexes, metadata, and registry from disk and memory.
        Always call this when changing embedding models.
        """
        for path in [
            self._faiss_path,
            self._splade_path,
            self._bm25_path,
            self._metadata_path,
            self._registry_path,
        ]:
            if path.exists():
                path.unlink()

        self._metadata    = []
        self._splade_vecs = []
        self._faiss_index = None
        self._bm25._index = None
        self.registry     = {}
        print("Vector DB wiped. Run `pipeline.py index` to reindex from scratch.")

    # ------------------------------------------------------------------
    # Indexing — dense and SPLADE
    # ------------------------------------------------------------------

    def add_to_dense_and_splade(self, chunks: list[dict[str, Any]]) -> None:
        """
        Encode and add chunks to FAISS and SPLADE.

        Args:
            chunks: List of chunk dicts from pdf_parser.parse_pdf().
                    Each chunk must have: text, pdf_id, source, last_modified.
        """
        if not chunks:
            return

        texts    = [c["text"] for c in chunks]
        start_id = len(self._metadata)

        # Dense (FAISS)
        print(f"  Dense-embedding {len(texts)} chunks ...")
        dense_vecs = _l2_normalise(self._dense.embed(texts))

        dim = dense_vecs.shape[1]
        if self._faiss_index is None:
            self._faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

        ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)
        self._faiss_index.add_with_ids(dense_vecs, ids) # type: ignore

        # SPLADE
        print(f"  SPLADE-encoding {len(texts)} chunks ...")
        self._splade_vecs.extend(self._splade.encode(texts))

        # Metadata
        self._metadata.extend(chunks)

        # Registry — derive from chunk fields, deduplicate by pdf_id
        for chunk in chunks:
            self.registry[chunk["pdf_id"]] = {
                "source":        chunk["source"],
                "last_modified": chunk["last_modified"],
            }

        print(f"  Added {len(chunks)} chunks.")

    def remove_from_dense_and_splade(self, pdf_ids: list[str]) -> None:
        """
        Remove all chunks belonging to the given PDFs from FAISS, SPLADE,
        and metadata in one pass, with a single FAISS rebuild at the end.

        Args:
            pdf_ids: List of PDF identifiers to purge.
        """
        pdf_ids_set = set(pdf_ids)

        remove_set = {
            i for i, m in enumerate(self._metadata)
            if m["pdf_id"] in pdf_ids_set
        }
        if not remove_set:
            return

        print(f"  Removing {len(remove_set)} chunks for {len(pdf_ids)} PDF(s) ...")
        # Compact metadata and SPLADE in one pass
        keep              = [i for i in range(len(self._metadata)) if i not in remove_set]
        self._metadata    = [self._metadata[i]    for i in keep]
        self._splade_vecs = [self._splade_vecs[i] for i in keep]

        # Remove from FAISS
        d = self._faiss_index.d # type: ignore
        kept_faiss_vectors = np.empty((len(keep), d), dtype="float32")
        for j, i in enumerate(keep):
            kept_faiss_vectors[j] = self._faiss_index.reconstruct(i) # type: ignore

        start_id = len(self._metadata)
        dim = kept_faiss_vectors.shape[1]
        self._faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

        ids = np.arange(start_id, start_id + kept_faiss_vectors.shape[0], dtype=np.int64)
        self._faiss_index.add_with_ids(kept_faiss_vectors, ids) # type: ignore

        for pdf_id in pdf_ids:
            self.registry.pop(pdf_id, None)

    # ------------------------------------------------------------------
    # Indexing — BM25
    # ------------------------------------------------------------------

    def build_bm25(self) -> None:
        """
        Build the BM25 index from the current metadata corpus.
        Call this once at the end of a pipeline run, after all
        add_to_dense_and_splade() and remove_from_dense_and_splade() calls.
        """
        print(f"Building BM25 index over {len(self._metadata)} chunks ...")
        self._bm25.build_index([m["text"] for m in self._metadata])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: RewrittenQuery,
        k:     int | None = None,
        mode:  str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search and return the top-k most relevant chunks.

        Args:
            query: Natural-language query string.
            k:     Number of results (defaults to config value).
            mode:  "dense" | "sparse" | "hybrid" (defaults to config value).
                   For "sparse", which index is used is set by
                   config["retrieval"]["sparse_model"] ("bm25" or "splade").

        Returns:
            List of chunk dicts ordered best-first, each with a "score" key.
        """
        cfg          = self._config["retrieval"]
        k            = k    or cfg["k"]
        mode         = mode or cfg["mode"]
        rrf_constant = int(cfg["rrf_constant"])
        candidate_k  = max(k * 2, k + 20) # type: ignore

        if not self._metadata:
            return []

        if mode == "dense":
            scores, ids = self._dense_search(query, k) # type: ignore
        elif mode == "sparse":
            scores, ids =  self._sparse_search(query, k) # type: ignore
        elif mode == "hybrid":
            scores, ids = self._hybrid_search(query, k, candidate_k, rrf_constant) # type: ignore
        else:
            raise ValueError(f"Unknown retrieval mode: '{mode}'")
        return [
                {**self._metadata[idx], "score": score}
                for score, idx in zip(scores, ids)
            ]

    # ------------------------------------------------------------------
    # Internal search methods
    # ------------------------------------------------------------------

    def _dense_search(self, query: RewrittenQuery, k: int) -> tuple[list[float], list[int]]:
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return ([], [])

        q_vec       = _l2_normalise(self._dense.embed([query.dense]))
        scores, ids = self._faiss_index.search(
            q_vec, min(k, self._faiss_index.ntotal)
        )
        return (scores[0].tolist(), ids[0].tolist())

    def _sparse_search(self, query: RewrittenQuery, k: int) -> tuple[list[float], list[int]]:
        """Route to BM25 or SPLADE based on config."""
        sparse_model = self._config["retrieval"].get("sparse_model", "bm25")

        if sparse_model == "bm25":
            return self._bm25_search(query.bm25, k)
        elif sparse_model == "splade":
            return self._splade_search(query.splade, k)
        else:
            raise ValueError(f"Unknown sparse_model: '{sparse_model}'")

    def _bm25_search(self, query: str, k: int) -> tuple[list[float], list[int]]:
        scores = self._bm25.get_scores(query)
        if scores.size == 0:
            return ([], [])
        top_indices = np.argsort(scores)[::-1][:k]
        return ([float(scores[i]) for i in top_indices ], top_indices.tolist())

    def _splade_search(self, query: str, k: int) -> tuple[list[float], list[int]]:
        if not self._splade_vecs:
            return ([], [])
        q_vec  = self._splade.encode_one(query)
        scores = [
            SpladeEncoder.similarity(q_vec, doc_vec)
            for doc_vec in self._splade_vecs
        ]
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return ([float(scores[i]) for i in top_indices], top_indices)

    def _hybrid_search(
        self, query: RewrittenQuery, k: int, candidate_k: int, rrf_constant: int
    ) -> tuple[list[float], list[int]]:
        """
        Reciprocal Rank Fusion over dense and sparse candidate lists.
        RRF score = 1 / (rank_dense + C) + 1 / (rank_sparse + C)
        """
        _, d_ids  = self._dense_search(query, candidate_k)
        _, s_ids = self._sparse_search(query, candidate_k)

        rrf_scores: dict[int, float] = {}

        for rank, idx in enumerate(d_ids):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + _rrf_score(rank, rrf_constant)

        for rank, idx in enumerate(s_ids):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + _rrf_score(rank, rrf_constant)

        top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return ([s for _, s in top], [i for i, _ in top])