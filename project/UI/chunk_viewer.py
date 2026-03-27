"""
UI/chunk_viewer.py
------------------
Expandable chunk cards shown to the RIGHT of each assistant message.
Cited chunks are visually highlighted.

Layout:
  - Summary line: "📚 10 chunks · 3 cited"  (always visible, compact)
  - One ChunkCard per chunk:
      • Header: rank / source / page / score  (1 line, always visible)
      • Body:   full plain text, no internal scroll — expands to fit all content
  - Cards are stacked in a plain VBox inside a QScrollArea so the whole
    right-column panel can scroll if there are many chunks.

Pure display widget — no src imports.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)


class ChunkCard(QFrame):
    def __init__(self, chunk: dict, index: int, cited: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("ChunkCard")
        self.setProperty("cited", "true" if cited else "false")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header (always visible, 1 line) ─────────────────────────────
        header = QWidget()
        header.setObjectName("ChunkHeader")
        header.setProperty("cited", "true" if cited else "false")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(8, 5, 8, 5)
        hl.setSpacing(4)

        icon = QLabel("📄")
        icon.setFixedWidth(18)

        source = chunk.get("source", "unknown")
        page   = chunk.get("metadata", {}).get("page_number", "?")
        rank   = chunk.get("rank", index)
        score  = chunk.get("score", 0.0)

        info = QLabel(f"#{rank}  {source}  · p.{page}  · {score:.3f}")
        info.setObjectName("ChunkInfo")
        info.setProperty("cited", "true" if cited else "false")
        info.setFont(QFont("Consolas", 8))

        hl.addWidget(icon)
        hl.addWidget(info, 1)

        if cited:
            badge = QLabel("cited")
            badge.setObjectName("CitedBadge")
            hl.addWidget(badge)

        self._toggle_btn = QPushButton("▶")
        self._toggle_btn.setObjectName("ChunkToggle")
        self._toggle_btn.setFixedWidth(28)
        self._toggle_btn.clicked.connect(self._toggle)
        hl.addWidget(self._toggle_btn)

        layout.addWidget(header)

        # ── Body: plain QLabel, word-wrapped, no internal scroll ─────────
        self._body = QWidget()
        self._body.setObjectName("ChunkBody")
        bl = QVBoxLayout(self._body)
        bl.setContentsMargins(10, 6, 10, 10)
        bl.setSpacing(0)

        self._text_label = QLabel(chunk.get("text", ""))
        self._text_label.setObjectName("ChunkText")
        self._text_label.setWordWrap(True)
        self._text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._text_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,   # grows to fit all content — no scroll needed
        )
        self._text_label.setFont(QFont("Consolas", 8))
        bl.addWidget(self._text_label)

        self._body.setVisible(False)
        layout.addWidget(self._body)

        # Force style refresh after dynamic property set
        for w in (self, header, info):
            w.style().unpolish(w)
            w.style().polish(w)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._toggle_btn.setText("▼" if self._expanded else "▶")


class ChunkViewer(QWidget):
    def __init__(self, chunks: list[dict], cited_ids: list | None = None, parent=None):
        super().__init__(parent)
        cited_set = set(cited_ids or [])
        str_cited = {str(x) for x in cited_set}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        n = len(chunks)
        n_cited = sum(1 for c in chunks if str(c.get("rank")) in str_cited)

        summary = QLabel(
            f"📚  {n} chunk{'s' if n != 1 else ''}"
            + (f"  ·  {n_cited} cited" if n_cited else "")
        )
        summary.setObjectName("ChunksLabel")
        outer.addWidget(summary)

        # Outer scrollable area — cards themselves expand freely
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 2, 0, 2)
        cl.setSpacing(4)

        for i, chunk in enumerate(chunks, 1):
            rank = chunk.get("rank", i)
            cited = str(rank) in str_cited
            cl.addWidget(ChunkCard(chunk, i, cited=cited))

        cl.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll, 1)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)