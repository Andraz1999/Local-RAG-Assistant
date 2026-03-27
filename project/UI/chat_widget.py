"""
UI/chat_widget.py
-----------------
The main chat panel.

ARCHITECTURE NOTE:
  This widget never imports from src/ directly.
  It emits query_submitted when the user sends a message.
  main.py catches that, runs the RAG pipeline, then calls
  deliver_answer() or deliver_error() to push results back.

Conversation model (updated):
  Each Q+A pair is its own conversation — when a new question arrives,
  the current conversation is saved and a fresh one is created.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)

from UI.chunk_viewer import ChunkViewer
from UI.conversation import add_message, save_conversation, new_conversation


class UserBubble(QFrame):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("UserBubble")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(60, 8, 12, 8)
        label = QLabel(text)
        label.setObjectName("UserBubbleText")
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)


class AssistantBubble(QFrame):
    """
    Left column: answer text.
    Right column: ChunkViewer (if chunks provided).
    """
    def __init__(self, text: str, chunks: list | None = None,
                 cited_ids: list | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("AssistantBubble")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(12)

        # ── Answer text (left) ──────────────────────────────────────────
        answer_widget = QWidget()
        al = QVBoxLayout(answer_widget)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(0)

        label = QLabel(text)
        label.setObjectName("AssistantBubbleText")
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        al.addWidget(label)

        outer.addWidget(answer_widget, 3)   # takes 3/4 of width

        # ── Chunk viewer (right) ────────────────────────────────────────
        if chunks:
            chunk_widget = ChunkViewer(chunks, cited_ids=cited_ids or [])
            chunk_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            outer.addWidget(chunk_widget, 2)  # takes 2/4 of width


class ThinkingBubble(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ThinkingBubble")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 60, 8)
        self._label = QLabel("Thinking ·")
        self._label.setObjectName("ThinkingText")
        layout.addWidget(self._label)
        layout.addStretch()
        self._dots = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(400)

    def _animate(self) -> None:
        self._dots = (self._dots % 3) + 1
        self._label.setText("Thinking " + "·" * self._dots)

    def stop(self) -> None:
        self._timer.stop()


class ChatWidget(QWidget):
    query_submitted = pyqtSignal(str)   # → main.py runs the pipeline

    def __init__(self, parent=None):
        super().__init__(parent)
        self._conversation = new_conversation()
        self._thinking_bubble: ThinkingBubble | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setObjectName("ChatScroll")
        layout.addWidget(self._scroll, 1)

        self._messages_widget = QWidget()
        self._messages_widget.setObjectName("MessagesWidget")
        self._messages_layout = QVBoxLayout(self._messages_widget)
        self._messages_layout.setContentsMargins(8, 12, 8, 8)
        self._messages_layout.setSpacing(6)
        self._messages_layout.addStretch()
        self._scroll.setWidget(self._messages_widget)

        input_frame = QFrame()
        input_frame.setObjectName("InputFrame")
        il = QHBoxLayout(input_frame)
        il.setContentsMargins(12, 10, 12, 10)
        il.setSpacing(8)

        self._input = QTextEdit()
        self._input.setObjectName("ChatInput")
        self._input.setPlaceholderText("Ask a question about your documents…")
        self._input.setFixedHeight(108)  # ~5 lines

        send_btn = QPushButton("Send  ↵")
        send_btn.setObjectName("SendBtn")
        send_btn.setFixedSize(88, 108)
        send_btn.clicked.connect(self._send)
        QShortcut(QKeySequence("Ctrl+Return"), self._input).activated.connect(self._send)

        il.addWidget(self._input, 1)
        il.addWidget(send_btn)
        layout.addWidget(input_frame)

    # ── Public API (called by main.py) ──────────────────────────────────

    def load_conversation(self, conv: dict) -> None:
        """Load a saved conversation for display (read-only replay)."""
        self._conversation = conv
        self._clear_messages()
        for msg in conv.get("messages", []):
            if msg["role"] == "user":
                self._add_user_bubble(msg["content"])
            else:
                self._add_assistant_bubble(
                    msg["content"],
                    msg.get("chunks"),
                    msg.get("cited_ids", []),
                )

    def start_new_conversation(self) -> None:
        self._conversation = new_conversation()
        self._clear_messages()

    def deliver_answer(self, answer_text: str, chunks: list, cited_ids: list) -> None:
        self._remove_thinking()
        self._add_assistant_bubble(answer_text, chunks or None, cited_ids)
        add_message(self._conversation, "assistant", answer_text,
                    chunks or None, cited_ids or None)
        save_conversation(self._conversation)
        # ── Each Q+A is its own conversation: start fresh for the next one ──
        self._conversation = new_conversation()
        self._input.setEnabled(True)
        self._input.setFocus()

    def deliver_error(self, error_msg: str) -> None:
        self._remove_thinking()
        self._add_assistant_bubble(f"⚠  {error_msg}")
        self._input.setEnabled(True)
        self._input.setFocus()

    def get_conversation_id(self) -> str:
        return self._conversation["id"]

    # ── Private ─────────────────────────────────────────────────────────

    def _send(self) -> None:
        query = self._input.toPlainText().strip()
        if not query:
            return
        self._input.clear()
        # Start a fresh conversation for this Q+A pair
        self._conversation = new_conversation()
        self._clear_messages()
        self._add_user_bubble(query)
        add_message(self._conversation, "user", query)
        self._add_thinking()
        self._input.setEnabled(False)
        self.query_submitted.emit(query)

    def _clear_messages(self) -> None:
        while self._messages_layout.count() > 1:
            item = self._messages_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

    def _add_user_bubble(self, text: str) -> None:
        self._messages_layout.addWidget(UserBubble(text))
        self._scroll_to_bottom()

    def _add_assistant_bubble(self, text: str, chunks: list | None = None,
                               cited_ids: list | None = None) -> None:
        self._messages_layout.addWidget(AssistantBubble(text, chunks, cited_ids))
        self._scroll_to_bottom()

    def _add_thinking(self) -> None:
        self._thinking_bubble = ThinkingBubble()
        self._messages_layout.addWidget(self._thinking_bubble)
        self._scroll_to_bottom()

    def _remove_thinking(self) -> None:
        if self._thinking_bubble:
            self._thinking_bubble.stop()
            self._messages_layout.removeWidget(self._thinking_bubble)
            self._thinking_bubble.deleteLater()
            self._thinking_bubble = None

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(50, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        ))