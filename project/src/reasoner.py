"""
reasoner.py
-----------
Answers a user's question based on the context and data that it gets.

Usage:

"""
import textwrap
import re
from dataclasses import dataclass, field
from typing import Optional, Any
import httpx # type: ignore

from src.config_loader import get_reasoner_model
from .query_rewriter import RewrittenQuery
        
GENERATION_OPTIONS = {
    "temperature": 0.0,    # low = more factual / deterministic
    "num_predict": 2048,   # max tokens for the answer
    "num_ctx":     8192,   # context window; reduce to 4096 if RAM is tight
}

NO_INFO_MESSAGE = (
    "There is no information available in the retrieved chunks to answer this question. "
    "Try changing your settings or increasing k."
)

@dataclass
class RAGAnswer:
    question:      str
    answer:        str
    cited_ids:     list[int | str]
    rewritten:     Optional[RewrittenQuery] = None
    chunks_used:   list[dict[str, Any]] = field(default_factory=list)
    raw_response:  str = ""
 
    def __str__(self) -> str:
        divider = "─" * 60
        lines = [
            divider,
            f"Question : {self.question}",
            divider,
            self.answer,
            "",
            f"Cited chunk IDs : {self.cited_ids}",
        ]
        if self.chunks_used:
            lines.append("\nSources:")
            for chunk in self.chunks_used:
                lines.append(f"Result {chunk['rank']}  |  source: {chunk['source']}  | page number: {chunk['metadata']['page_number']}  |  score: {chunk['score']:.4f}")
        lines.append(divider)
        return "\n".join(lines)



# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
You are a precise Retrieval-Augmented Generation (RAG) assistant.
 
Your task:
- Answer the user's question STRICTLY using the information in the provided chunks.
- Do NOT use any external knowledge or make things up.
- If the chunks do not contain enough information, reply EXACTLY with:
  {NO_INFO_MESSAGE} \
- Be concise and factual. Prefer bullet points for lists.
- At the END of your answer, on its own line, write the IDs of every chunk you used,
  in this exact format:   CITED_CHUNKS: [1, 2, 3]
  If you used no chunks (no information case), write:  CITED_CHUNKS: []
"""

def _build_user_prompt(query: str, chunks: list[dict[str, Any]], intent: str = "") -> str:
    """Render the retrieval context + question into the user turn."""
    chunk_block = "\n\n".join(
        f"--- CHUNK {chunk['rank']}  |  source: {chunk['source']}  | page number: {chunk['metadata']['page_number']}  |  score: {chunk['score']:.4f} ---\n{chunk['text']}"
        for chunk in chunks
    )
 
    intent_line = f"\nSearch intent: {intent}\n" if intent else ""
 
    return textwrap.dedent(f"""\
        {intent_line}
        === RETRIEVED CHUNKS ===
        {chunk_block}
 
        === QUESTION ===
        {query}
    """).strip()

def _parse_response(raw: str) -> tuple[str, list[int | str]]:
    """
    Split the model response into:
      - answer text  (everything before CITED_CHUNKS line)
      - list of cited IDs
 
    Also strips <think>...</think> blocks produced by DeepSeek-R1.
    """
    # Remove reasoning traces from DeepSeek-R1
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
 
    # Extract CITED_CHUNKS: [...]
    cited_ids: list[int | str] = []
    cite_match = re.search(r"CITED_CHUNKS:\s*(\[.*?\])", cleaned, re.IGNORECASE)
    if cite_match:
        try:
            raw_ids = cite_match.group(1)
            # Parse mixed int/str IDs safely
            cited_ids = [
                int(x) if x.strip().isdigit() else x.strip().strip("'\"")
                for x in raw_ids.strip("[]").split(",")
                if x.strip()
            ]
        except Exception:
            cited_ids = []
        # Remove the CITED_CHUNKS line from the answer
        answer = cleaned[: cite_match.start()].strip()
    else:
        answer = cleaned.strip()
 
    return answer, cited_ids


# ── Reasoning / Answering ────────────────────────────────────────────────────────────

def answer(query: RewrittenQuery, chunks: list[dict[str, Any]], config: dict[str, Any]) -> RAGAnswer:
    """
    Answers a RAG query using provided chunks of text.

    Args:
        query: A RewrittenQuery object with a query for the dense, splade and bm25 model
        chunks: The text where the answer of the query should lie
        config:   Loaded project config.

    Returns:
        RAGAnswer Object:
            (
            question,
            answer,
            cited_ids,
            rewritten,
            chunks_used,
            raw_response,
        )

    """
    intent = query.intent if query.intent else ""
    user_prompt = _build_user_prompt(query.original, chunks, intent)
 
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
 
    try:
        model = get_reasoner_model(config)
        client = httpx.Client(base_url=config["embedding"].get("ollama_base_url", "http://localhost:11434"), timeout=600)
        response = client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": GENERATION_OPTIONS
            }
        )
        response.raise_for_status()
        resp_json = response.json()
        
        # Handle different response formats
        if "message" in resp_json and "content" in resp_json["message"]:
            raw = resp_json["message"]["content"].strip()
        elif "response" in resp_json:
            raw = resp_json["response"].strip()
        else:
            raw = NO_INFO_MESSAGE
    except Exception as e:
        raw = f"Error calling reasoning model: {str(e)}. Retrieved chunks:\n\n" + "\n\n".join(
            f"[{chunk['rank']}] {chunk['text'][:200]}..."
            for chunk in chunks[:3]
        )
    
    answer_text, cited_ids = _parse_response(raw)
 
    chunks_used = []
    for chunk in chunks:
        if chunk["rank"] in cited_ids:
            chunks_used.append(chunk) 
 
    return RAGAnswer(
        question=query.original,
        answer=answer_text,
        cited_ids=cited_ids,
        rewritten=query,
        chunks_used=chunks_used,
        raw_response=raw,
    )