"""
query_rewriter.py
-----------------
Transforms a raw user query into multiple optimised search queries
suitable for hybrid retrieval (dense vectors + BM25 + SPLADE).

Model: 
       Change REWRITER_MODEL to any Ollama model you prefer.

Usage:
    from query_rewriter import rewrite_query

    results = rewrite_query("What are the side effects of ibuprofen?")
    # results is a RewrittenQuery dataclass
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Any

import httpx  # type: ignore
from src.config_loader import get_rewriter_model

# ── Configuration ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a search query optimisation assistant for a Retrieval-Augmented Generation (RAG) system.
Your job is to take a user's raw question and produce MULTIPLE reformulated search queries that
together maximise recall across different retrieval methods:
  - Dense vector search  (semantic similarity)
  - BM25 keyword search  (exact term matching)
  - SPLADE sparse vectors (expanded term matching)

Rules:
1. Produce exactly 3 query variants as a JSON object with keys:
   - "dense"  : a natural-language paraphrase, enriched with synonyms and context
   - "bm25"   : a keyword-focused query (important nouns, verbs, no filler words)
   - "splade" : a term-expanded query with related concepts and alternative phrasings
2. Also include:
   - "intent"  : one sentence describing what the user is actually looking for
   - "filters" : a list of any explicit constraints (date, source, page, etc.), empty list if none
3. Return ONLY valid JSON, no markdown, no explanation.

Example output:
{
  "dense":  "What are the known adverse reactions and health risks associated with ibuprofen use?",
  "bm25":   "ibuprofen side effects risks adverse reactions",
  "splade": "ibuprofen NSAID side effects gastrointestinal bleeding kidney damage cardiovascular risk overdose",
  "intent": "The user wants a list of potential negative health effects from taking ibuprofen.",
  "filters": []
}
"""

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RewrittenQuery:
    original: str
    dense:    str
    bm25:     str
    splade:   str
    intent:   str
    filters:  list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "original": self.original,
            "dense":    self.dense,
            "bm25":     self.bm25,
            "splade":   self.splade,
            "intent":   self.intent,
            "filters":  self.filters,
        }

    def __str__(self) -> str:
        lines = [
            f"Original : {self.original}",
            f"Intent   : {self.intent}",
            f"Dense    : {self.dense}",
            f"BM25     : {self.bm25}",
            f"SPLADE   : {self.splade}",
        ]
        if self.filters:
            lines.append(f"Filters  : {', '.join(self.filters)}")
        return "\n".join(lines)


# ── Core function ─────────────────────────────────────────────────────────────

def rewrite_query(
    query: str,
    config: dict[str, Any],
    fallback_on_error: bool = True,
) -> RewrittenQuery:
    """
    Send the raw query to the local Ollama model and parse the JSON response.

    Parameters
    ----------
    query            : The user's original question or search string.
    config           : Loaded project config
    fallback_on_error: If True, returns a passthrough RewrittenQuery on parse
                       failure instead of raising an exception.

    Returns
    -------
    RewrittenQuery dataclass with all variants populated.
    """

    model  = get_rewriter_model(config)
    if(model == "Disabled"):
        return _passthrough(query)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Rewrite this query:\n\n{query}"},
    ]

    try:
        client     = httpx.Client(base_url=config["embedding"].get("ollama_base_url", "http://localhost:11434"), timeout=120)
        response = client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {     
                    "temperature": 0.2,
                    "num_predict": 512,
                    "num_ctx": 8192,
                }
            }
)
        raw_text = response.json()["message"]["content"].strip()
        parsed   = _parse_json_response(raw_text)

        return RewrittenQuery(
            original=query,
            dense=parsed.get("dense",  query),
            bm25=parsed.get("bm25",   query),
            splade=parsed.get("splade", query),
            intent=parsed.get("intent", ""),
            filters=parsed.get("filters", []),
        )

    except Exception as exc:
        if fallback_on_error:
            print(f"[query_rewriter] Warning: rewriting failed ({exc}). Using original query.")
            return _passthrough(query)
        raise


def _parse_json_response(text: str) -> dict:
    """Extract and parse the first JSON object found in the model output."""
    # Strip markdown code fences if the model added them
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: grab the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"No valid JSON found in model output:\n{text}")


def _passthrough(query: str) -> RewrittenQuery:
    """Return the original query unchanged for all variants."""
    return RewrittenQuery(
        original=query,
        dense=query,
        bm25=query,
        splade=query,
        intent="(query rewriting unavailable — using original)",
        filters=[],
    )

