"""
config_loader.py
----------------
Loads and validates the project config.json.
Import load_config() in any script that needs settings.
"""

import json
from pathlib import Path
from typing import Any

_REQUIRED_KEYS = {
    "paths":     ["input_folder", "db_folder"],
    "partition": ["strategy"],
    "chunking":  ["strategy", "max_characters", "new_after_n_chars", "overlap"],
    "embedding": [
        "ollama_embedding_model_names",
        "hf_embedding_model_names",
        "current_embedding_model",
        "splade_model_names",
        "current_splade_model",
        "batch_size",
    ],
    "retrieval": ["mode", "k", "rrf_constant"],
    "rewriter":  ["model_names", "current_rewriter_model"],
    "reasoner":  ["model_names", "current_reasoner_model"],
}

_VALID_RETRIEVAL_MODES       = {"dense", "sparse", "hybrid"}
_VALID_CHUNKING_STRATEGIES   = {"by_title", "basic"}
_VALID_PARTITION_STRATEGIES  = {"fast", "auto", "hi_res", "ocr_only"}
_VALID_SPARSE_MODELS         = {"bm25", "splade"}


def load_config(config_path: str | Path = "../config.json") -> dict[str, Any]:
    """
    Load and validate config.json.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Validated config dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required keys are missing or values are invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    _validate(config)
    return config


def _validate(config: dict[str, Any]) -> None:
    # ── Required keys ────────────────────────────────────────────────────
    for section, keys in _REQUIRED_KEYS.items():
        if section not in config:
            raise ValueError(f"Config missing section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Config missing key: '{section}.{key}'")

    # ── Retrieval mode ───────────────────────────────────────────────────
    mode = config["retrieval"]["mode"]
    if mode not in _VALID_RETRIEVAL_MODES:
        raise ValueError(
            f"Invalid retrieval.mode '{mode}'. Must be one of: {_VALID_RETRIEVAL_MODES}"
        )

    # ── Sparse model tag ─────────────────────────────────────────────────
    sparse_model = config["retrieval"].get("sparse_model", "bm25")
    if sparse_model not in _VALID_SPARSE_MODELS:
        raise ValueError(
            f"Invalid retrieval.sparse_model '{sparse_model}'. "
            f"Must be one of: {_VALID_SPARSE_MODELS}"
        )
    config["retrieval"]["sparse_model"] = sparse_model

    # ── Chunking strategy ────────────────────────────────────────────────
    chunking_strategy = config["chunking"]["strategy"]
    if chunking_strategy not in _VALID_CHUNKING_STRATEGIES:
        raise ValueError(
            f"Invalid chunking.strategy '{chunking_strategy}'. "
            f"Must be one of: {_VALID_CHUNKING_STRATEGIES}"
        )

    # ── Partition strategy ───────────────────────────────────────────────
    partition_strategy = config["partition"]["strategy"]
    if partition_strategy not in _VALID_PARTITION_STRATEGIES:
        raise ValueError(
            f"Invalid partition.strategy '{partition_strategy}'. "
            f"Must be one of: {_VALID_PARTITION_STRATEGIES}"
        )

    # ── k ────────────────────────────────────────────────────────────────
    k = config["retrieval"]["k"]
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"retrieval.k must be a positive integer, got: {k}")

    # ── Embedding model index ────────────────────────────────────────────
    emb       = config["embedding"]
    ollama_models = emb.get("ollama_embedding_model_names", [])
    hf_models     = emb.get("hf_embedding_model_names", [])

    if not isinstance(ollama_models, list) or not isinstance(hf_models, list):
        raise ValueError(
            "embedding.ollama_embedding_model_names and "
            "embedding.hf_embedding_model_names must be lists."
        )

    total_embedding_models = len(ollama_models) + len(hf_models)
    if total_embedding_models == 0:
        raise ValueError("At least one embedding model must be defined.")

    current_emb = emb["current_embedding_model"]
    if not isinstance(current_emb, int) or not (0 <= current_emb < total_embedding_models):
        raise ValueError(
            f"embedding.current_embedding_model ({current_emb}) is out of range. "
            f"Must be 0–{total_embedding_models - 1} "
            f"(ollama: {len(ollama_models)}, hf: {len(hf_models)})."
        )

    # ── SPLADE model index ───────────────────────────────────────────────
    splade_models = emb.get("splade_model_names", [])
    if not isinstance(splade_models, list) or len(splade_models) == 0:
        raise ValueError("embedding.splade_model_names must be a non-empty list.")

    current_splade = emb["current_splade_model"]
    if not isinstance(current_splade, int) or not (0 <= current_splade < len(splade_models)):
        raise ValueError(
            f"embedding.current_splade_model ({current_splade}) is out of range. "
            f"Must be 0–{len(splade_models) - 1}."
        )

    # ── Rewriter model index ─────────────────────────────────────────────
    rewriter_models = config["rewriter"].get("model_names", [])
    if not isinstance(rewriter_models, list):
        raise ValueError("rewriter.model_names must be a list (can be empty to disable).")

    current_rewriter = config["rewriter"]["current_rewriter_model"]
    # -1 means disabled; otherwise must be a valid index
    if current_rewriter != -1:
        if not isinstance(current_rewriter, int) or not (0 <= current_rewriter < len(rewriter_models)):
            raise ValueError(
                f"rewriter.current_rewriter_model ({current_rewriter}) is out of range. "
                f"Use -1 to disable or 0–{max(0, len(rewriter_models) - 1)}."
            )

    # ── Reasoner model index ─────────────────────────────────────────────
    reasoner_models = config["reasoner"].get("model_names", [])
    if not isinstance(reasoner_models, list) or len(reasoner_models) == 0:
        raise ValueError("reasoner.model_names must be a non-empty list.")

    current_reasoner = config["reasoner"]["current_reasoner_model"]
    if not isinstance(current_reasoner, int) or not (0 <= current_reasoner < len(reasoner_models)):
        raise ValueError(
            f"reasoner.current_reasoner_model ({current_reasoner}) is out of range. "
            f"Must be 0–{len(reasoner_models) - 1}."
        )


# ---------------------------------------------------------------------------
# Convenience helpers (used by src/ modules)
# ---------------------------------------------------------------------------

def get_embedding_model(config: dict) -> tuple[str, str]:
    """
    Returns (provider, model_name) for the currently selected embedding model.
    provider is 'ollama' or 'huggingface'.
    """
    emb           = config["embedding"]
    ollama_models = emb["ollama_embedding_model_names"]
    hf_models     = emb["hf_embedding_model_names"]
    idx           = emb["current_embedding_model"]

    if idx < len(ollama_models):
        return "ollama", ollama_models[idx]
    else:
        return "huggingface", hf_models[idx - len(ollama_models)]


def get_splade_model(config: dict) -> str:
    """Returns the currently selected SPLADE model name."""
    emb = config["embedding"]
    return emb["splade_model_names"][emb["current_splade_model"]]


def get_rewriter_model(config: dict) -> str | None:
    """Returns the rewriter model name, or None if disabled."""
    rw = config["rewriter"]
    idx = rw["current_rewriter_model"]
    if idx == -1 or not rw["model_names"]:
        return None
    return rw["model_names"][idx]


def get_reasoner_model(config: dict) -> str:
    """Returns the reasoner model name."""
    rs = config["reasoner"]
    return rs["model_names"][rs["current_reasoner_model"]]