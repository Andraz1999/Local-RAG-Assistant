"""
get_models.py
Helper script called by start.bat to extract Ollama model names from config.json.
Prints one model name per line.
"""
import json
import sys

config_path = sys.argv[1] if len(sys.argv) > 1 else "project/config.json"

try:
    with open(config_path, encoding="utf-8") as f:
        c = json.load(f)

    emb             = c.get("embedding", {})
    ollama_models   = emb.get("ollama_embedding_model_names", [])
    rewriter_models = [m for m in c.get("rewriter", {}).get("model_names", []) if m != "Disabled"]
    reasoner_models = c.get("reasoner", {}).get("model_names", [])

    to_pull = ollama_models + rewriter_models + reasoner_models

    seen = set()
    for m in to_pull:
        if m not in seen:
            print(m)
            seen.add(m)

except Exception as e:
    print(f"CONFIG_ERROR: {e}", file=sys.stderr)
    sys.exit(1)