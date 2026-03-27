"""
parse_worker.py
---------------
Standalone script invoked by main.py via subprocess.run() to parse a single PDF.
Running outside of Qt entirely avoids any thread/process conflicts with unstructured.

Usage (internal — called by _IndexWorker):
    python parse_worker.py <pdf_path> <config_json_path> <output_json_path>

Writes a JSON array of chunk dicts to output_json_path on success.
Writes nothing and exits with code 1 on failure (error goes to stderr).
"""

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: parse_worker.py <pdf_path> <config_json> <output_json>", file=sys.stderr)
        sys.exit(1)

    pdf_path    = Path(sys.argv[1])
    config_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # src/ must be importable — this script sits at the project root, same as main.py
    from src.pdf_parser import parse_pdf
    chunks = parse_pdf(pdf_path, config)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)


if __name__ == "__main__":
    main()