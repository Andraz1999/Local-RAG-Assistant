"""
UI/history_panel.py
-------------------
Left sidebar listing saved conversations.
Pure display widget — no src imports.
"""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMenu, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from UI.conversation import list_conversations, delete_conversation


class HistoryPanel(QWidget):
    conversation_selected      = pyqtSignal(str)   # conv_id
    new_conversation_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HistoryPanel")
        self.setFixedWidth(240)
        self._setup_ui()
        self.refresh()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(8)

        title = QLabel("Conversations")
        title.setObjectName("PanelTitle")
        title.setFont(QFont("Segoe UI Semibold", 11))
        layout.addWidget(title)

        new_btn = QPushButton("＋  New conversation")
        new_btn.setObjectName("NewConvBtn")
        new_btn.clicked.connect(self.new_conversation_requested)
        layout.addWidget(new_btn)

        self._list = QListWidget()
        self._list.setObjectName("HistoryList")
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._context_menu)
        self._list.itemClicked.connect(self._on_click)
        layout.addWidget(self._list)

    def refresh(self) -> None:
        self._list.clear()
        for conv in list_conversations():
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, conv["id"])
            try:
                dt = datetime.fromisoformat(conv["created_at"])
                date_str = dt.strftime("%Y %b %d, %H:%M")
            except Exception:
                date_str = ""
            widget = _ConvItem(conv["title"], date_str)
            item.setSizeHint(widget.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, widget)

    def _on_click(self, item: QListWidgetItem) -> None:
        conv_id = item.data(Qt.ItemDataRole.UserRole)
        if conv_id:
            self.conversation_selected.emit(conv_id)

    def _context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if not item:
            return
        conv_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        delete_action = menu.addAction("🗑  Delete")
        action = menu.exec(self._list.mapToGlobal(pos))
        if action == delete_action:
            delete_conversation(conv_id)
            self.refresh()


class _ConvItem(QWidget):
    def __init__(self, title: str, date: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("ConvTitle")
        t.setWordWrap(False)
        d = QLabel(date)
        d.setObjectName("ConvDate")
        layout.addWidget(t)
        layout.addWidget(d)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)