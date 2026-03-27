"""
UI/conversation.py
------------------
JSON save/load for conversations.
File I/O only — no src imports.

Each conversation JSON:
  { id, title, created_at, messages: [{role, content, timestamp, chunks?, cited_ids?}] }

Each conversation is a single Q+A pair (question + answer + chunks).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

CONVERSATIONS_DIR = Path(__file__).parent / "conversations"


def _ensure_dir() -> None:
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)


def new_conversation() -> dict:
    return {
        "id":         str(uuid.uuid4()),
        "title":      "New conversation",
        "created_at": datetime.now().isoformat(),
        "messages":   [],
    }


def save_conversation(conv: dict) -> None:
    _ensure_dir()
    path = CONVERSATIONS_DIR / f"{conv['id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)


def load_conversation(conv_id: str) -> dict | None:
    path = CONVERSATIONS_DIR / f"{conv_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_conversations() -> list[dict]:
    _ensure_dir()
    convs = []
    for p in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            convs.append({
                "id":         data["id"],
                "title":      data.get("title", "Untitled"),
                "created_at": data.get("created_at", ""),
            })
        except Exception:
            pass
    convs.sort(key=lambda c: c["created_at"], reverse=True)
    return convs


def delete_conversation(conv_id: str) -> None:
    path = CONVERSATIONS_DIR / f"{conv_id}.json"
    if path.exists():
        path.unlink()


def add_message(conv: dict, role: str, content: str,
                chunks: list | None = None,
                cited_ids: list | None = None) -> dict:
    msg: dict = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
    if chunks:
        msg["chunks"] = chunks
    if cited_ids:
        msg["cited_ids"] = cited_ids
    conv["messages"].append(msg)
    if role == "user" and conv["title"] == "New conversation":
        conv["title"] = content[:60] + ("…" if len(content) > 60 else "")
    return msg