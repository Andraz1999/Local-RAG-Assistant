"""
pipeline.py
-----------
Main entry point for the RAG pipeline.

Commands:
    python pipeline.py index                          — parse PDFs, update all indexes
    python pipeline.py search "my query"              — search with config defaults
    python pipeline.py search "my query" --k 10      — override k
    python pipeline.py search "my query" --mode dense — override retrieval mode
    python pipeline.py reset                          — wipe entire vector DB
"""


from __future__ import annotations

import argparse

import sys
sys.path.append("..")

from src.config_loader import load_config
from src.embedder import get_encoders
from src.pdf_parser import get_pdf_fingerprint, get_pdf_updates, make_pdf_id, parse_pdf
from src.vector_store import VectorStore
from src.query_rewriter import RewrittenQuery, rewrite_query
from src.reasoner import answer



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_store(config: dict) -> VectorStore:
    """Instantiate all encoders and return a loaded VectorStore."""
    dense, splade, bm25 = get_encoders(config)
    store = VectorStore(config, dense, splade, bm25)
    store.load()
    return store


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def run_index(config: dict) -> None:
    """
    Incrementally update the vector DB:
      1. Parse chunks for all new and changed PDFs
      2. Delete changed and removed PDFs from dense + SPLADE
      3. Add all new chunks at once to dense + SPLADE
      4. Build BM25 once from the final corpus
      5. Save
    """
    store = _build_store(config)
 
    print("\nScanning input folder …")
    updates = get_pdf_updates(config, store.registry)
 
    has_changes = any([updates["to_delete"], updates["to_reindex"], updates["to_index"]])
    if not has_changes:
        print("Nothing to index — all PDFs are up to date.")
        return

    # Step 1 — Parse all PDFs that need (re)indexing
    print("\nParsing PDFs ...")
    chunks_to_add: list[dict] = []
    i = 0
    for pdf_path in updates["to_reindex"] + updates["to_index"]:
        i+=1
        print(f" {i}/{len(updates["to_reindex"] + updates["to_index"])} Parsing {pdf_path.name} ...")
        try:
            chunks = parse_pdf(pdf_path, config)
            if chunks:
                chunks_to_add.extend(chunks)
            else:
                print(f"  Warning: no chunks extracted from {pdf_path.name}")
        except Exception as e:
            print(f"  Error parsing {pdf_path.name}: {e}")

    # Step 2 — Remove deleted and changed PDFs from dense + SPLADE
    print(f"\nRemoving {len(updates["to_reindex"] + updates["to_delete"])} deleted and changed PDFs ...")
    store.remove_from_dense_and_splade(updates["to_reindex"] + updates["to_delete"])

    # Step 3 — Add new chunks
    if chunks_to_add:
        print(f"\nAdding {len(chunks_to_add)} chunks to database ...")
        store.add_to_dense_and_splade(chunks_to_add)

    # Step 4 — Build BM25 once from the final corpus state
    print("\nBuilding BM25 ...")
    store.build_bm25()

    # Step 5 — Save everything
    print("\nSaving ...")
    store.save()
    print("\nDone.")


def run_reset(config: dict) -> None:
    """Wipe the entire vector DB (required when changing embedding models)."""
    store = _build_store(config)
    store.reset()


def run_query(
    config: dict,
    query:  str,
    k:      int | None = None,
    mode:   str | None = None,
) -> None:
    """Search the vector store and print results."""
    store = _build_store(config)

    if not store.registry:
        print("Vector DB is empty. Run `python pipeline.py index` first.")
        return

    k    = k    or config["retrieval"]["k"]
    mode = mode or config["retrieval"]["mode"]

    if config["rewriter"]["model"] != "":
        print("Rewriting your query")
    
    rewritten = rewrite_query(query, config)

    print(f"Query for dense: {rewritten.dense}")
    print("---")
    print(f"Query for bm25: {rewritten.bm25}")
    print("---")
    print(f"Query for splade: {rewritten.splade}")
    print("---")
    print(f"User's intent: {rewritten.intent}")
    print("---")


    print(f'\nSearching (mode={mode}, k={k}): "{query}"\n')
    results = store.search(rewritten, k=k, mode=mode)

    


    if not results:
        print("No results found.")
        return

    for i, chunk in enumerate(results, 1):
        chunk["rank"] = i
        print(f"{'─' * 60}")
        print(f"Result {i}  |  source: {chunk['source']}  | page number: {chunk["metadata"]["page_number"]}  |  score: {chunk['score']:.4f}")
        print(f"{'─' * 60}")
        print(chunk["text"])
        print()


    print("Answering your query...")
    rag_answer = answer(rewritten, results, config)
    print(rag_answer)

    


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PDF RAG pipeline")
    parser.add_argument(
        "--config", default="../config.json", help="Path to config.json"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Parse PDFs and update the vector DB")
    subparsers.add_parser("reset", help="Wipe the entire vector DB")

    search_parser = subparsers.add_parser("search", help="Query the vector DB")
    search_parser.add_argument("query", help="Query string")
    search_parser.add_argument(
        "--k", type=int, default=None, help="Number of results to return"
    )
    search_parser.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid"],
        default=None,
        help="Retrieval mode (overrides config)",
    )

    args   = parser.parse_args()
    config = load_config(args.config)

    if args.command == "index":
        run_index(config)
    elif args.command == "reset":
        run_reset(config)
    elif args.command == "search":
        run_query(config, args.query, k=args.k, mode=args.mode)


if __name__ == "__main__":
    main()