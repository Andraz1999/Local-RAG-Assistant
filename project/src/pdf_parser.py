"""
pdf_parser.py
-------------
Parses PDFs in the configured input folder into chunks using `unstructured`.

Tracks which PDFs have already been processed (by filename + last-modified
timestamp) so only new or changed files are re-parsed on subsequent runs.
Deleted PDFs are flagged so their vectors can be removed from the DB.

Typical usage:
    from pdf_parser import get_pdf_updates, parse_pdf

    updates = get_pdf_updates(config, registry)
    for pdf_path in updates["to_index"]:
        chunks = parse_pdf(pdf_path, config)
"""

import hashlib
import os
from pathlib import Path
from typing import Any



# ---------------------------------------------------------------------------
# Registry helpers  (registry = dict persisted in vector_db/registry.json)
# ---------------------------------------------------------------------------

def make_pdf_id(pdf_path: Path) -> str:
    """
    Stable ID for a PDF: MD5 of its absolute path string.
    Using the path (not content) keeps IDs stable for rename-detection,
    while last_modified catches content changes.
    """
    return hashlib.md5(str(pdf_path.resolve()).encode()).hexdigest()


def get_pdf_fingerprint(pdf_path: Path) -> dict[str, Any]:
    """Return a dict with the file's name and last-modified timestamp."""
    stat = pdf_path.stat()
    return {
        "name": pdf_path.name,
        "last_modified": stat.st_mtime,
    }


def get_pdf_updates(
    config: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, list]:
    """
    Compare PDFs on disk against the registry.

    Args:
        config:   Loaded project config.
        registry: Currently persisted registry (may be empty dict).

    Returns:
        {
            "to_index":  [Path, ...],  # new or changed PDFs to parse + index
            "to_reindex": to_reindex, # changed files (need delete + add)
            "to_delete": [str, ...],   # pdf_ids removed from disk
        }
    """
    input_folder = Path(config["paths"]["input_folder"])
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    all_pdf_paths = list(input_folder.rglob("*.pdf"))

    disk_pdfs: dict[str, Path] = {
        make_pdf_id(p): p for p in all_pdf_paths
    }

    to_index: list[Path] = []
    to_reindex: list[Path] = []
    to_delete: list[str] = []

    # Detect new or changed
    for pdf_id, pdf_path in disk_pdfs.items():
        fp = get_pdf_fingerprint(pdf_path)
        if pdf_id not in registry:
            print(f"  [NEW]     {pdf_path.name}")
            to_index.append(pdf_path)
        elif registry[pdf_id]["last_modified"] != fp["last_modified"]:
            print(f"  [CHANGED] {pdf_path.name}")
            to_reindex.append(pdf_path)
        else:
            print(f"  [SKIP]    {pdf_path.name}  (unchanged)")

    # Detect deleted
    for pdf_id in registry:
        if pdf_id not in disk_pdfs:
            print(f"  [DELETED] {registry[pdf_id]['source']}")
            to_delete.append(pdf_id)

    return {"to_index": to_index, "to_reindex": to_reindex, "to_delete": to_delete}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Partition and chunk a single PDF.

    Args:
        pdf_path: Path to the PDF file.
        config:   Loaded project config.

    Returns:
        List of chunk dicts:
        {
            "text":          str,
            "source":        str,   # filename
            "pdf_id":        str,   # stable PDF identifier
            "chunk_index":   int,   # position within this PDF
            "last_modified": float, # PDF mtime
            "metadata":      dict,  # unstructured element metadata
        }
    """
    

    print(f"got {pdf_path}")
    from unstructured.chunking.basic import chunk_elements # type: ignore
    from unstructured.chunking.title import chunk_by_title # type: ignore
    from unstructured.partition.pdf import partition_pdf # type: ignore

    print("imports")

    partition_cfg = config["partition"]
    chunking_cfg = config["chunking"]

    import os
    os.environ["UNSTRUCTURED_PARALLEL_MODE_ENABLED"] = "false"

    print("read cfg")

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy=partition_cfg["strategy"],
        include_metadata=True,
    )

    print("make element")

    if chunking_cfg["strategy"] == "by_title":
        chunks = chunk_by_title(
            elements,
            max_characters=chunking_cfg["max_characters"],
            new_after_n_chars=chunking_cfg["new_after_n_chars"],
            overlap=chunking_cfg["overlap"],
        )
    else:
        chunks = chunk_elements(
            elements,
            max_characters=chunking_cfg["max_characters"],
            overlap=chunking_cfg["overlap"],
        )

    print("chunk finished")
    

    pdf_id = make_pdf_id(pdf_path)
    last_modified = pdf_path.stat().st_mtime

    return [
        {
            "text": chunk.text,
            "source": pdf_path.name,
            "pdf_id": pdf_id,
            "chunk_index" : idx,
            "last_modified": last_modified,
            "metadata": (
                chunk.metadata.to_dict()
                if hasattr(chunk.metadata, "to_dict")
                else {}
            ),
        }
        for idx, chunk in enumerate(chunks)
        if chunk.text.strip()  # drop empty chunks
    ]

