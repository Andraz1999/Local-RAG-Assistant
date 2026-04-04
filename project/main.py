"""
main.py  (project root)
-----------------------
Single entry point for everything.

  python main.py          → launch the GUI
  python main.py index    → index PDFs (CLI, no GUI)
  python main.py search "query"
  python main.py reset

Because this file sits at the project root, Python adds the root to
sys.path automatically.  So:
  from src.embedder import ...    ← works
  from UI.chat_widget import ...  ← works
No sys.path tricks needed anywhere.

Architecture:
  main.py owns the QApplication and the RAG pipeline.
  UI widgets emit signals → main.py calls src/ → pushes results back to UI.
  UI widgets never import from src/ directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal as Signal, Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QProgressBar,
    QStatusBar, QVBoxLayout, QWidget,
)
from PyQt6.QtGui import QIcon

# src imports — work because main.py is at the project root
from src.config_loader import load_config as _load_config_from_file
from src.embedder     import get_encoders
from src.pdf_parser   import get_pdf_updates, parse_pdf
from src.vector_store import VectorStore
from src.query_rewriter import rewrite_query
from src.reasoner     import answer as rag_answer

# UI imports — also work for the same reason
from UI.chat_widget    import ChatWidget
from UI.history_panel  import HistoryPanel
from UI.settings_panel import SettingsDialog, DEFAULT_VISUAL
from UI.conversation   import load_conversation

# For start.bat to work
import sys, os
if sys.stdout is None: sys.stdout = open(os.devnull, 'w')
if sys.stderr is None: sys.stderr = open(os.devnull, 'w')
if sys.stdin is None: sys.stdin = open(os.devnull, 'r')


# ---------------------------------------------------------------------------
# Stylesheet templates  (dark + light, {fs} placeholders for font size)
# ---------------------------------------------------------------------------

_DARK = """
QWidget {{
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: "Segoe UI", "SF Pro Text", sans-serif;
    font-size: {fs}pt;
}}
/* ── Scrollbars: dark track, light handle ── */
QScrollBar:vertical {{
    background: #1e2235; width: 8px; border-radius: 4px;
}}
QScrollBar:horizontal {{
    background: #1e2235; height: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: #6366f1; border-radius: 4px; min-height: 30px; min-width: 30px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: #1e2235;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    height: 0; width: 0;
}}

/* Disable mouse-wheel on combo boxes */
QComboBox {{ qproperty-focusPolicy: StrongFocus; }}

#HeaderBar {{
    background-color: #13161f;
    border-bottom: 1px solid #252836;
    min-height: 56px; max-height: 56px;
}}
#AppTitle {{
    color: #a5b4fc; font-size: {fs_large}pt; font-weight: 700;
}}
#HeaderSubtitle {{ color: #64748b; font-size: {fs_small}pt; }}
#HeaderBtn {{
    background: #1e2235; color: #94a3b8;
    border: 1px solid #2d3250; border-radius: 6px;
    padding: 7px 14px;
}}
#HeaderBtn:hover {{ background: #252a3d; color: #c7d2fe; border-color: #4f5a8a; }}
#HeaderBtn:checked {{ background: #2d3561; color: #a5b4fc; }}

/* Soft purple-ish right border on history panel */
#HistoryPanel {{ background-color: #13161f; border-right: 2px solid #7c6fcf; }}
#PanelTitle {{ color: #a5b4fc; font-size: {fs_medium}pt; font-weight: 700; padding: 4px 0; }}
#NewConvBtn {{
    background: #1e2235; color: #a5b4fc;
    border: 1px solid #3d4475; border-radius: 6px;
    padding: 8px 10px; text-align: left;
}}
#NewConvBtn:hover {{ background: #252a3d; border-color: #6366f1; }}
#HistoryList {{ background: transparent; border: none; outline: none; }}
#HistoryList::item {{ border-radius: 6px; padding: 2px; margin: 1px 0; }}
#HistoryList::item:selected {{ background: #1e2549; }}
#HistoryList::item:hover {{ background: #1a1e30; }}
#ConvTitle {{ color: #e2e8f0; }}
#ConvDate  {{ color: #4a5568; font-size: {fs_small}pt; }}

#ChatScroll, #MessagesWidget {{ background: #0f1117; }}
#UserBubble {{
    background-color: #1e2549; border: 1px solid #2d3561;
    border-radius: 12px; margin: 2px 0;
}}
#UserBubbleText {{ color: #c7d2fe; line-height: 1.5; }}
#AssistantBubble {{
    background-color: #161b2e; border: 1px solid #252836;
    border-radius: 12px; margin: 2px 0;
}}
#AssistantBubbleText {{ color: #e2e8f0; line-height: 1.6; }}
#ThinkingBubble {{
    background-color: #161b2e; border: 1px dashed #2d3561; border-radius: 12px;
}}
#ThinkingText {{ color: #6366f1; font-style: italic; }}
#InputFrame {{ background-color: #13161f; border-top: 1px solid #252836; }}
#ChatInput {{
    background: #1a1d27; color: #e2e8f0;
    border: 1px solid #2d3250; border-radius: 8px; padding: 8px 12px;
}}
#ChatInput:focus {{ border-color: #6366f1; }}
#ChatInput:disabled {{ color: #4a5568; }}
#SendBtn {{
    background: #4f46e5; color: #fff;
    border: none; border-radius: 8px; font-weight: 600;
}}
#SendBtn:hover {{ background: #6366f1; }}
#SendBtn:disabled {{ background: #2d3250; color: #4a5568; }}

#ChunksLabel {{ color: #64748b; font-size: {fs_small}pt; padding: 4px 2px 2px; }}
#ChunkCard {{ background: #12151f; border: 1px solid #1e2235; border-radius: 6px; }}
#ChunkCard[cited="true"] {{ border: 1px solid #6366f1; background: #131828; }}
#ChunkHeader {{ background: #181c2e; border-radius: 6px 6px 0 0; }}
#ChunkHeader[cited="true"] {{ background: #1a2040; }}
#ChunkInfo {{ color: #64748b; font-size: {fs_small}pt; }}
#ChunkInfo[cited="true"] {{ color: #a5b4fc; }}
#CitedBadge {{
    background: #312e81; color: #a5b4fc;
    border: 1px solid #4338ca; border-radius: 3px;
    padding: 1px 5px; font-size: {fs_small}pt; font-weight: 600;
}}
#ChunkToggle {{
    background: #1e2235; color: #94a3b8;
    border: 1px solid #2d3250; border-radius: 4px; padding: 3px 6px;
    font-size: {fs_small}pt;
}}
#ChunkToggle:hover {{ background: #252a3d; color: #c7d2fe; }}
#ChunkText {{
    background: #0d1018; color: #94a3b8;
    border: 1px solid #1e2235; border-radius: 4px;
    font-family: "Consolas", "Fira Code", monospace;
    font-size: {fs_small}pt;
    padding: 6px 8px;
}}

#SettingsDialog {{ background: #0f1117; }}
#DialogTitle {{ color: #a5b4fc; font-size: {fs_xlarge}pt; font-weight: 700; }}
#SettingsTabs {{ background: #0f1117; border: none; }}
#SettingsTabs::pane {{
    border: 1px solid #252836; border-radius: 8px; background: #13161f;
}}
#SettingsTabs QTabBar::tab {{
    background: #1a1d27; color: #64748b;
    border: 1px solid #252836; border-bottom: none;
    border-radius: 6px 6px 0 0; padding: 10px 20px; margin-right: 3px;
}}
#SettingsTabs QTabBar::tab:selected {{ background: #13161f; color: #a5b4fc; border-color: #3d4475; }}
#SettingsTabs QTabBar::tab:hover:!selected {{ background: #1e2235; color: #94a3b8; }}
QGroupBox#SettingsGroup {{
    color: #64748b; font-size: {fs_small}pt; font-weight: 700; letter-spacing: 0.8px;
    border: 1px solid #1e2235; border-radius: 8px;
    margin-top: 14px; padding-top: 8px;
}}
QGroupBox#SettingsGroup::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 0 8px; left: 10px; color: #94a3b8;
}}
#FieldLabel {{ color: #94a3b8; }}
#WarnBanner {{
    background: #2d1a0e; color: #fb923c;
    border: 1px solid #7c2d12; border-radius: 6px; padding: 10px 14px;
}}
#WarnBannerEmbedding {{
    background: #1a1535; color: #a5b4fc;
    border: 1px solid #4338ca; border-radius: 6px; padding: 10px 14px;
}}
#FontPreview {{ color: #e2e8f0; padding: 8px 4px; }}
#SliderLabel {{ color: #64748b; font-size: {fs_small}pt; }}
#FontSizeLabel {{ color: #a5b4fc; font-weight: 700; }}
#ThemeChoiceBtn {{
    background: #1a1d27; color: #94a3b8;
    border: 2px solid #2d3250; border-radius: 8px; font-size: {fs_medium}pt; padding: 10px;
}}
#ThemeChoiceBtn:hover {{ border-color: #4f5a8a; color: #c7d2fe; }}
#ThemeChoiceBtn:checked {{ background: #1e2549; color: #a5b4fc; border-color: #6366f1; }}
#HelpIcon {{color: #64748b; font-weight: 700; padding-left: 2px; }}
#HelpIcon:hover {{color: #6366f1; }}
QLineEdit, QComboBox, QSpinBox {{
    background: #1a1d27; color: #e2e8f0;
    border: 1px solid #2d3250; border-radius: 5px; padding: 6px 10px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border-color: #6366f1; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: #1a1d27; border: 1px solid #2d3250;
    selection-background-color: #2d3561; color: #e2e8f0;
}}
QSpinBox::up-button, QSpinBox::down-button {{ background: #252a3d; border: none; width: 20px; }}
#SmallBtn {{
    background: #1e2235; color: #94a3b8;
    border: 1px solid #2d3250; border-radius: 5px; padding: 6px;
}}
#SmallBtn:hover {{ background: #252a3d; color: #c7d2fe; }}
#ScanBtn {{
    background: #1e2549; color: #a5b4fc;
    border: 1px solid #3d4475; border-radius: 6px; padding: 8px 14px; font-weight: 600;
}}
#ScanBtn:hover {{ background: #252a3d; border-color: #6366f1; }}
#RebuildBtn {{
    background: #3b1f10; color: #fb923c;
    border: 1px solid #7c2d12; border-radius: 6px; padding: 8px 14px; font-weight: 600;
}}
#RebuildBtn:hover {{ background: #431f0e; border-color: #ea580c; }}
#ResetDbBtn {{
    background: #2d1535; color: #c084fc;
    border: 1px solid #6b21a8; border-radius: 6px; padding: 6px 12px; font-weight: 600;
}}
#ResetDbBtn:hover {{ background: #3b1a4a; border-color: #a855f7; color: #e9d5ff; }}
#OkBtn {{
    background: #4f46e5; color: #fff;
    border: 1px solid #4f46e5; border-radius: 6px; padding: 8px 18px;
    min-width: 70px; font-weight: 600;
}}
#OkBtn:hover {{ background: #6366f1; border-color: #6366f1; }}
QDialogButtonBox QPushButton {{
    background: #1e2235; color: #94a3b8;
    border: 1px solid #2d3250; border-radius: 6px; padding: 8px 18px; min-width: 70px;
}}
QDialogButtonBox QPushButton:hover {{ background: #252a3d; color: #c7d2fe; }}
QDialogButtonBox QPushButton[text="OK"] {{ background: #4f46e5; color: white; border-color: #4f46e5; }}
QDialogButtonBox QPushButton[text="OK"]:hover {{ background: #6366f1; }}
QSlider::groove:horizontal {{ background: #2d3250; height: 6px; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: #6366f1; width: 18px; height: 18px; border-radius: 9px; margin: -6px 0;
}}
QSlider::sub-page:horizontal {{ background: #6366f1; border-radius: 3px; }}
QStatusBar {{
    background: #0d0f17; color: #4a5568;
    border-top: 1px solid #1a1d27; font-size: {fs_small}pt;
}}
QProgressBar {{
    background: #1a1d27; border: 1px solid #2d3250; border-radius: 6px;
    text-align: center; color: #e2e8f0; height: 22px;
}}
QProgressBar::chunk {{
    background: #6366f1; border-radius: 5px;
}}
"""

_LIGHT = """
QWidget {{
    background-color: #f8fafc; color: #1e293b;
    font-family: "Segoe UI", "SF Pro Text", sans-serif; font-size: {fs}pt;
}}
/* ── Scrollbars: light track, dark handle ── */
QScrollBar:vertical {{
    background: #e2e8f0; width: 8px; border-radius: 4px;
}}
QScrollBar:horizontal {{
    background: #e2e8f0; height: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: #6366f1; border-radius: 4px; min-height: 30px; min-width: 30px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: #e2e8f0;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    height: 0; width: 0;
}}

/* Disable mouse-wheel on combo boxes */
QComboBox {{ qproperty-focusPolicy: StrongFocus; }}

#HeaderBar {{ background-color: #fff; border-bottom: 1px solid #e2e8f0; min-height: 56px; max-height: 56px; }}
#AppTitle {{ color: #4f46e5; font-size: {fs_large}pt; font-weight: 700; }}
#HeaderSubtitle {{ color: #94a3b8; font-size: {fs_small}pt; }}
#HeaderBtn {{
    background: #f1f5f9; color: #475569;
    border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px 14px;
}}
#HeaderBtn:hover {{ background: #e2e8f0; color: #1e293b; }}
#HeaderBtn:checked {{ background: #ede9fe; color: #4f46e5; border-color: #a5b4fc; }}

/* Soft purple right border on history panel */
#HistoryPanel {{ background-color: #fff; border-right: 2px solid #7c6fcf; }}
#PanelTitle {{ color: #4f46e5; font-size: {fs_medium}pt; font-weight: 700; padding: 4px 0; }}
#NewConvBtn {{
    background: #ede9fe; color: #4f46e5; border: 1px solid #a5b4fc;
    border-radius: 6px; padding: 8px 10px; text-align: left;
}}
#NewConvBtn:hover {{ background: #ddd6fe; }}
#HistoryList {{ background: transparent; border: none; outline: none; }}
#HistoryList::item {{ border-radius: 6px; padding: 2px; margin: 1px 0; }}
#HistoryList::item:selected {{ background: #ede9fe; }}
#HistoryList::item:hover {{ background: #f1f5f9; }}
#ConvTitle {{ color: #1e293b; }}
#ConvDate  {{ color: #94a3b8; font-size: {fs_small}pt; }}

#ChatScroll, #MessagesWidget {{ background: #f8fafc; }}
#UserBubble {{ background-color: #ede9fe; border: 1px solid #c4b5fd; border-radius: 12px; margin: 2px 0; }}
#UserBubbleText {{ color: #3730a3; line-height: 1.5; }}
#AssistantBubble {{ background-color: #fff; border: 1px solid #e2e8f0; border-radius: 12px; margin: 2px 0; }}
#AssistantBubbleText {{ color: #1e293b; line-height: 1.6; }}
#ThinkingBubble {{ background-color: #fff; border: 1px dashed #c4b5fd; border-radius: 12px; }}
#ThinkingText {{ color: #6366f1; font-style: italic; }}
#InputFrame {{ background-color: #fff; border-top: 1px solid #e2e8f0; }}
#ChatInput {{
    background: #f8fafc; color: #1e293b;
    border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px 12px;
}}
#ChatInput:focus {{ border-color: #6366f1; }}
#SendBtn {{ background: #4f46e5; color: #fff; border: none; border-radius: 8px; font-weight: 600; }}
#SendBtn:hover {{ background: #6366f1; }}
#SendBtn:disabled {{ background: #e2e8f0; color: #94a3b8; }}

#ChunksLabel {{ color: #94a3b8; font-size: {fs_small}pt; padding: 4px 2px 2px; }}
#ChunkCard {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; }}
#ChunkCard[cited="true"] {{ border: 1px solid #6366f1; background: #f0effe; }}
#ChunkHeader {{ background: #f1f5f9; border-radius: 6px 6px 0 0; }}
#ChunkHeader[cited="true"] {{ background: #ede9fe; }}
#ChunkInfo {{ color: #94a3b8; font-size: {fs_small}pt; }}
#ChunkInfo[cited="true"] {{ color: #4f46e5; }}
#CitedBadge {{
    background: #ede9fe; color: #4f46e5;
    border: 1px solid #a5b4fc; border-radius: 3px;
    padding: 1px 5px; font-size: {fs_small}pt; font-weight: 600;
}}
#ChunkToggle {{
    background: #f1f5f9; color: #64748b;
    border: 1px solid #cbd5e1; border-radius: 4px; padding: 3px 6px; font-size: {fs_small}pt;
}}
#ChunkToggle:hover {{ background: #e2e8f0; }}
#ChunkText {{
    background: #fff; color: #475569; border: 1px solid #e2e8f0; border-radius: 4px;
    font-family: "Consolas", "Fira Code", monospace; font-size: {fs_small}pt;
    padding: 6px 8px;
}}

#SettingsDialog {{ background: #f8fafc; }}
#DialogTitle {{ color: #4f46e5; font-size: {fs_xlarge}pt; font-weight: 700; }}
#SettingsTabs {{ background: #f8fafc; border: none; }}
#SettingsTabs::pane {{ border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; }}
#SettingsTabs QTabBar::tab {{
    background: #f1f5f9; color: #64748b; border: 1px solid #e2e8f0; border-bottom: none;
    border-radius: 6px 6px 0 0; padding: 10px 20px; margin-right: 3px;
}}
#SettingsTabs QTabBar::tab:selected {{ background: #fff; color: #4f46e5; border-color: #c4b5fd; }}
QGroupBox#SettingsGroup {{
    color: #64748b; font-size: {fs_small}pt; font-weight: 700; letter-spacing: 0.8px;
    border: 1px solid #e2e8f0; border-radius: 8px; margin-top: 14px; padding-top: 8px;
}}
QGroupBox#SettingsGroup::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 0 8px; left: 10px; color: #94a3b8;
}}
#FieldLabel {{ color: #64748b; }}
#WarnBanner {{
    background: #fff7ed; color: #c2410c;
    border: 1px solid #fed7aa; border-radius: 6px; padding: 10px 14px;
}}
#WarnBannerEmbedding {{
    background: #ede9fe; color: #4338ca;
    border: 1px solid #a5b4fc; border-radius: 6px; padding: 10px 14px;
}}
#FontPreview {{ color: #1e293b; padding: 8px 4px; }}
#SliderLabel {{ color: #94a3b8; font-size: {fs_small}pt; }}
#FontSizeLabel {{ color: #4f46e5; font-weight: 700; }}
#ThemeChoiceBtn {{
    background: #f8fafc; color: #64748b;
    border: 2px solid #e2e8f0; border-radius: 8px; font-size: {fs_medium}pt; padding: 10px;
}}
#ThemeChoiceBtn:hover {{ border-color: #a5b4fc; color: #4f46e5; }}
#ThemeChoiceBtn:checked {{ background: #ede9fe; color: #4f46e5; border-color: #6366f1; }}
#HelpIcon {{color: #64748b; font-weight: 700; padding-left: 2px; }}
#HelpIcon:hover {{color: #6366f1; }}
QLineEdit, QComboBox, QSpinBox {{
    background: #fff; color: #1e293b; border: 1px solid #cbd5e1;
    border-radius: 5px; padding: 6px 10px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border-color: #6366f1; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: #fff; border: 1px solid #cbd5e1;
    selection-background-color: #ede9fe; color: #1e293b;
}}
QSpinBox::up-button, QSpinBox::down-button {{ background: #f1f5f9; border: none; width: 20px; }}
#SmallBtn {{ background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; border-radius: 5px; padding: 6px; }}
#SmallBtn:hover {{ background: #e2e8f0; }}
#ScanBtn {{ background: #ede9fe; color: #4f46e5; border: 1px solid #a5b4fc; border-radius: 6px; padding: 8px 14px; font-weight: 600; }}
#ScanBtn:hover {{ background: #ddd6fe; }}
#RebuildBtn {{ background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; border-radius: 6px; padding: 8px 14px; font-weight: 600; }}
#RebuildBtn:hover {{ background: #ffedd5; }}
#ResetDbBtn {{ background: #faf5ff; color: #7e22ce; border: 1px solid #c084fc; border-radius: 6px; padding: 6px 12px; font-weight: 600; }}
#ResetDbBtn:hover {{ background: #f3e8ff; border-color: #a855f7; color: #6b21a8; }}
#OkBtn {{
    background: #4f46e5; color: #fff;
    border: 1px solid #4f46e5; border-radius: 6px; padding: 8px 18px;
    min-width: 70px; font-weight: 600;
}}
#OkBtn:hover {{ background: #6366f1; border-color: #6366f1; }}
QDialogButtonBox QPushButton {{
    background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1;
    border-radius: 6px; padding: 8px 18px; min-width: 70px;
}}
QDialogButtonBox QPushButton:hover {{ background: #e2e8f0; }}
QDialogButtonBox QPushButton[text="OK"] {{ background: #4f46e5; color: white; border-color: #4f46e5; }}
QDialogButtonBox QPushButton[text="OK"]:hover {{ background: #6366f1; }}
QSlider::groove:horizontal {{ background: #e2e8f0; height: 6px; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: #6366f1; width: 18px; height: 18px; border-radius: 9px; margin: -6px 0; }}
QSlider::sub-page:horizontal {{ background: #6366f1; border-radius: 3px; }}
QStatusBar {{ background: #f1f5f9; color: #94a3b8; border-top: 1px solid #e2e8f0; font-size: {fs_small}pt; }}
QProgressBar {{
    background: #e2e8f0; border: 1px solid #cbd5e1; border-radius: 6px;
    text-align: center; color: #1e293b; height: 22px;
}}
QProgressBar::chunk {{
    background: #6366f1; border-radius: 5px;
}}
"""

def _stylesheet(theme: str, fs: int) -> str:
    tpl = _DARK if theme == "dark" else _LIGHT
    return tpl.format(
        fs=fs,
        fs_small=max(fs - 2, 7),
        fs_medium=fs + 1,
        fs_large=fs + 4,
        fs_xlarge=fs + 6,
    )


# ---------------------------------------------------------------------------
# Error message helpers
# ---------------------------------------------------------------------------

def _friendly_error(exc: Exception) -> str:
    """
    Translate known low-level exceptions into user-readable messages.
    Falls back to the raw exception string for anything unrecognised.
    """
    msg = str(exc)
    print(msg)
    if "input length exceeds the context length" in msg:
        return (
            "The embedding model's context window is too small for one or more "
            "of your chunks.\n\n"
            "Try one of the following:\n"
            "  \u2022 Switch to a model with a larger context window (e.g. nomic-embed-text "
            "supports up to 8 192 tokens, while mxbai-embed-large only supports 512).\n"
            "  \u2022 Reduce the chunk size in Settings \u2192 Chunking "
            "(lower 'Max characters' and 'New after N chars').\n\n"
            "After changing the embedding model you must reset and re-index the database."
        )
    return msg


# ---------------------------------------------------------------------------
# Background worker for RAG queries
# ---------------------------------------------------------------------------

class _RAGWorker(QObject):
    finished = Signal(str, list, list)   # answer, chunks, cited_ids
    error    = Signal(str)

    def __init__(self, query: str, config: dict):
        super().__init__()
        self.query  = query
        self.config = config

    def run(self) -> None:
        try:
            dense, splade, bm25 = get_encoders(self.config)
            store = VectorStore(self.config, dense, splade, bm25)
            store.load()

            if not store.registry:
                self.error.emit("Vector DB is empty — use Settings → Scan to index your documents.")
                return

            rewritten = rewrite_query(self.query, self.config)
            
            print("---")
            print(f"Query for dense: {rewritten.dense}")
            print("---")
            print(f"Query for bm25: {rewritten.bm25}")
            print("---")
            print(f"Query for splade: {rewritten.splade}")
            print("---")
            print(f"User's intent: {rewritten.intent}")
            print("---")

            k    = self.config["retrieval"]["k"]
            mode = self.config["retrieval"]["mode"]
            results = store.search(rewritten, k=k, mode=mode)

            if not results:
                self.finished.emit("No relevant documents found for your query.", [], [])
                return

            for i, chunk in enumerate(results, 1):
                chunk["rank"] = i

            rag_answer_obj = rag_answer(rewritten, results, self.config)
            cited_ids = getattr(rag_answer_obj, "cited_ids", [])
            self.finished.emit(rag_answer_obj.answer, results, cited_ids)

        except Exception as exc:
            self.error.emit(_friendly_error(exc))


# ---------------------------------------------------------------------------
# Background worker for indexing — with progress signals
# ---------------------------------------------------------------------------

class _IndexWorker(QObject):
    progress      = Signal(int, int, str)   # current, total, phase
    done          = Signal(str)
    cancelled     = Signal()

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            self.progress.emit(0, 1, "Loading encoders…")
            dense, splade, bm25 = get_encoders(self.config)
            store = VectorStore(self.config, dense, splade, bm25)
            store.load()

            updates    = get_pdf_updates(self.config, store.registry)
            to_delete  = updates.get("to_delete", [])
            to_reindex = updates.get("to_reindex", [])
            to_index   = updates.get("to_index", [])

            if not any([to_delete, to_reindex, to_index]):
                self.done.emit("Nothing to index — all PDFs are up to date.")
                return

            pdfs_to_parse = to_reindex + to_index
            total_pdfs    = max(len(pdfs_to_parse), 1)
            total_steps   = total_pdfs * 2   # parse + embed phases
            chunks_to_add = []

            # ── Phase 1: Parse PDFs ──────────────────────────────────────
            # unstructured (partition_pdf) is incompatible with QThread —
            # it segfaults or hangs regardless of env-var workarounds.
            # Solution: invoke parse_worker.py as a completely separate
            # Python process for each PDF. No Qt, no threads, no conflicts.
            import os
            import subprocess
            import tempfile

            worker_script = Path(__file__).parent / "parse_worker.py"

            for i, pdf_path in enumerate(pdfs_to_parse):
                if self._cancel:
                    self.cancelled.emit(); return
                self.progress.emit(i, total_steps, f"Parsing {pdf_path.name}…")
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".json", delete=False, mode="w"
                    ) as cfg_f:
                        json.dump(self.config, cfg_f, ensure_ascii=False)
                        cfg_tmp = cfg_f.name

                    out_tmp = cfg_tmp + "_chunks.json"

                    result = subprocess.run(
                        [sys.executable, str(worker_script),
                         str(pdf_path), cfg_tmp, out_tmp],
                        capture_output=True, text=True
                    )

                    os.unlink(cfg_tmp)

                    if result.returncode != 0:
                        self.done.emit(
                            f"Parse error on {pdf_path.name}:\n"
                            f"{result.stderr or result.stdout}"
                        )
                        return

                    with open(out_tmp, "r", encoding="utf-8") as f:
                        chunks = json.load(f)
                    os.unlink(out_tmp)

                    if chunks:
                        chunks_to_add.extend(chunks)

                except Exception as parse_exc:
                    import traceback
                    self.done.emit(
                        f"Parse error on {pdf_path.name}:\n"
                        f"{parse_exc}\n{traceback.format_exc()}"
                    )
                    return

            if self._cancel:
                self.cancelled.emit(); return

            # ── Remove stale / deleted chunks ────────────────────────────
            ids_to_remove = [str(p) if not isinstance(p, str) else p
                             for p in to_reindex + to_delete]
            if ids_to_remove:
                store.remove_from_dense_and_splade(ids_to_remove)

            # ── Phase 2: Embed chunks ────────────────────────────────────
            total_chunks = len(chunks_to_add)
            if chunks_to_add:
                batch_size = self.config.get("embedding", {}).get("batch_size", 32)
                for i in range(0, total_chunks, batch_size):
                    if self._cancel:
                        self.cancelled.emit(); return
                    done_pdfs = total_pdfs   # parsing done
                    embed_step = int((i / total_chunks) * total_pdfs)
                    self.progress.emit(
                        done_pdfs + embed_step, total_steps,
                        f"Embedding chunks… ({min(i + batch_size, total_chunks)}/{total_chunks})"
                    )
                store.add_to_dense_and_splade(chunks_to_add)

            self.progress.emit(total_steps - 1, total_steps, "Building BM25 index…")
            store.build_bm25()
            store.save()
            self.done.emit(f"Indexing complete — {len(chunks_to_add)} chunks added ✓")

        except Exception as exc:
            import traceback
            self.done.emit(f"Indexing error: {_friendly_error(exc)}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Background worker for reset + full re-index (used after model change)
# ---------------------------------------------------------------------------

class _ResetAndIndexWorker(_IndexWorker):
    """Wipes the vector DB first, then indexes all PDFs from scratch."""

    def run(self) -> None:
        try:
            self.progress.emit(0, 1, "Resetting vector database…")
            dense, splade, bm25 = get_encoders(self.config)
            store = VectorStore(self.config, dense, splade, bm25)
            store.load()
            store.reset()
            self.progress.emit(0, 1, "Database wiped — starting full index…")
        except Exception as exc:
            import traceback
            self.done.emit(f"Reset error: {exc}\n{traceback.format_exc()}")
            return
        # Delegate to normal index flow — registry is empty so all PDFs are "new"
        super().run()


# ---------------------------------------------------------------------------
# Indexing progress dialog
# ---------------------------------------------------------------------------

class IndexProgressDialog(QDialog):
    cancel_requested = Signal()

    def __init__(self, parent=None, title: str = "Scanning Database",
                 heading: str = "📂  Caching the database…"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title_lbl = QLabel(heading)
        title_lbl.setFont(QFont("Segoe UI Semibold", 12))
        layout.addWidget(title_lbl)

        self._status = QLabel("Preparing…")
        self._status.setObjectName("FieldLabel")
        layout.addWidget(self._status)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        layout.addWidget(self._bar)

        btn_row = QHBoxLayout()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._ok_btn = QPushButton("OK")
        self._ok_btn.setObjectName("OkBtn")
        self._ok_btn.clicked.connect(self.accept)
        self._ok_btn.hide()
        btn_row.addStretch()
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._ok_btn)
        layout.addLayout(btn_row)

    def update_progress(self, current: int, total: int, msg: str) -> None:
        pct = int(current / total * 100) if total > 0 else 0
        self._bar.setValue(pct)
        self._status.setText(msg)

    def mark_done(self, msg: str) -> None:
        self._bar.setValue(100)
        self._status.setText(f"✓  {msg}")
        self._cancel_btn.hide()
        self._ok_btn.show()

    def mark_cancelled(self) -> None:
        self._status.setText("Cancelled.")
        self._cancel_btn.hide()
        self._ok_btn.show()

    def _on_cancel(self) -> None:
        self._status.setText("Cancelling…")
        self._cancel_btn.setEnabled(False)
        self.cancel_requested.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, config: dict, visual: dict):
        super().__init__()
        self._config = config
        self._visual = visual
        self._rag_thread: QThread | None = None
        self.setWindowTitle("RAG Assistant")
        self.setMinimumSize(800, 560)
        self.resize(1100, 760)
        self._setup_ui()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QFrame()
        header.setObjectName("HeaderBar")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)
        hl.setSpacing(10)

        tc = QVBoxLayout(); tc.setSpacing(1)
        t = QLabel("RAG Assistant"); t.setObjectName("AppTitle")
        s = QLabel("Retrieval-Augmented Generation · Local"); s.setObjectName("HeaderSubtitle")
        tc.addWidget(t); tc.addWidget(s)
        hl.addLayout(tc)
        hl.addStretch()

        self._hist_btn = QPushButton("☰  History")
        self._hist_btn.setObjectName("HeaderBtn")
        self._hist_btn.setCheckable(True)
        self._hist_btn.setChecked(True)
        self._hist_btn.clicked.connect(self._toggle_history)

        settings_btn = QPushButton("⚙  Settings")
        settings_btn.setObjectName("HeaderBtn")
        settings_btn.clicked.connect(self._open_settings)

        hl.addWidget(self._hist_btn)
        hl.addWidget(settings_btn)
        root.addWidget(header)

        # Body
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, 1)

        self._history = HistoryPanel()
        self._history.conversation_selected.connect(self._load_conv)
        self._history.new_conversation_requested.connect(self._new_conv)

        self._chat = ChatWidget()
        self._chat.query_submitted.connect(self._on_query)

        body.addWidget(self._history)
        body.addWidget(self._chat, 1)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._update_status()

    # ── History ──────────────────────────────────────────────────────────

    def _toggle_history(self) -> None:
        self._history.setVisible(self._hist_btn.isChecked())

    def _new_conv(self) -> None:
        self._chat.start_new_conversation()
        self._history.refresh()

    def _load_conv(self, conv_id: str) -> None:
        conv = load_conversation(conv_id)
        if conv:
            self._chat.load_conversation(conv)

    # ── Settings ─────────────────────────────────────────────────────────

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._config, self._visual, parent=self)
        dlg.settings_applied.connect(self._apply_settings)
        dlg.rebuild_requested.connect(self._on_rebuild)
        dlg.reset_requested.connect(self._on_rebuild)   # same flow: save + wipe + re-index
        dlg.index_requested.connect(self._on_index)
        dlg.exec()

    def _apply_settings(self, config: dict, visual: dict) -> None:
        self._config = config
        self._visual = visual
        QApplication.instance().setStyleSheet(_stylesheet(visual["theme"], visual["font_size"]))
        self._save_config(config)
        self._save_visual(visual)
        self._update_status()

    def _on_rebuild(self, config: dict, visual: dict) -> None:
        """Save advanced settings (including new model), then reset DB and re-index."""
        self._apply_settings(config, visual)
        self._run_index_worker(worker_class=_ResetAndIndexWorker,
                               title="Rebuilding Database",
                               heading="🔄  Resetting and rebuilding the database…")

    # ── RAG query (signal from ChatWidget) ───────────────────────────────

    def _on_query(self, query: str) -> None:
        if self._rag_thread and self._rag_thread.isRunning():
            return   # already processing

        self._rag_thread = QThread()
        worker = _RAGWorker(query, self._config)
        worker.moveToThread(self._rag_thread)
        self._rag_thread.started.connect(worker.run)
        worker.finished.connect(self._on_answer)
        worker.error.connect(self._on_error)
        worker.finished.connect(self._rag_thread.quit)
        worker.error.connect(self._rag_thread.quit)
        self._rag_worker = worker
        self._rag_thread.start()

    def _on_answer(self, answer_text: str, chunks: list, cited_ids: list) -> None:
        self._chat.deliver_answer(answer_text, chunks, cited_ids)
        self._history.refresh()

    def _on_error(self, msg: str) -> None:
        self._chat.deliver_error(msg)

    # ── Indexing ─────────────────────────────────────────────────────────

    def _on_index(self) -> None:
        self._run_index_worker()

    def _run_index_worker(self, worker_class=None, title: str = "Scanning Database",
                          heading: str = "📂  Caching the database…") -> None:
        if worker_class is None:
            worker_class = _IndexWorker

        dlg = IndexProgressDialog(self, title=title, heading=heading)
        self._idx_progress_dlg = dlg

        thread = QThread(self)           # parent=self keeps it alive
        worker = worker_class(self._config)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(dlg.update_progress)
        worker.done.connect(self._on_index_done)
        worker.done.connect(thread.quit)
        worker.cancelled.connect(self._on_index_cancelled)
        worker.cancelled.connect(thread.quit)
        dlg.cancel_requested.connect(worker.cancel)

        # Keep references so GC doesn't destroy them
        self._idx_thread = thread
        self._idx_worker = worker

        thread.start()
        dlg.exec()   # blocks until OK / auto-close on cancel

    def _on_index_done(self, msg: str) -> None:
        self._update_status(msg)
        if hasattr(self, "_idx_progress_dlg"):
            self._idx_progress_dlg.mark_done(msg)

    def _on_index_cancelled(self) -> None:
        self._status.showMessage("Indexing cancelled.")
        if hasattr(self, "_idx_progress_dlg"):
            self._idx_progress_dlg.mark_cancelled()

    # ── Persistence ──────────────────────────────────────────────────────

    def _save_config(self, config: dict) -> None:
        try:
            with open(Path(__file__).parent / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._status.showMessage(f"Could not save config: {e}")

    def _save_visual(self, visual: dict) -> None:
        try:
            with open(Path(__file__).parent / "UI" / "visual.json", "w", encoding="utf-8") as f:
                json.dump(visual, f, indent=2)
        except Exception:
            pass

    def _update_status(self, msg: str = "") -> None:
        if msg:
            self._status.showMessage(msg); return
        mode  = self._config.get("retrieval", {}).get("mode", "hybrid").upper()
        theme = self._visual.get("theme", "dark")
        fs    = self._visual.get("font_size", 10)
        self._status.showMessage(f"{mode} mode  ·  {theme} theme  ·  {fs}pt")


# ---------------------------------------------------------------------------
# CLI commands (no GUI)
# ---------------------------------------------------------------------------

def cli_index(config: dict) -> None:
    from src.pipeline import run_index
    run_index(config)


def cli_reset(config: dict) -> None:
    from src.pipeline import run_reset
    run_reset(config)


def cli_search(config: dict, query: str, k: int | None, mode: str | None) -> None:
    from src.pipeline import run_query
    run_query(config, query, k, mode)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_visual() -> dict:
    path = Path(__file__).parent / "UI" / "visual.json"
    if path.exists():
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    return dict(DEFAULT_VISUAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Assistant")
    parser.add_argument("--config", default="config.json")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("index")
    sub.add_parser("reset")
    sp = sub.add_parser("search")
    sp.add_argument("query")
    sp.add_argument("--k",    type=int, default=None)
    sp.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default=None)

    args   = parser.parse_args()
    config = _load_config_from_file(args.config)

    if args.cmd == "index":
        cli_index(config)
    elif args.cmd == "reset":
        cli_reset(config)
    elif args.cmd == "search":
        cli_search(config, args.query, args.k, args.mode)
    else:
        # No subcommand → launch GUI

        # Resolve icon path relative to this file so it works regardless of cwd
        _here = Path(__file__).parent
        _icon_path = str(_here / "icon.png")

        if sys.platform == "win32":
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RAGAssistant.App.1")
        elif sys.platform.startswith("linux"):
            # Tell the desktop environment which .desktop file owns this window.
            # This makes GNOME/KDE/etc. show your icon in the taskbar/dock
            # instead of a generic placeholder.
            # The value must match the filename of your .desktop file
            # (e.g. "rag-assistant" → /usr/share/applications/rag-assistant.desktop
            #  or   ~/.local/share/applications/rag-assistant.desktop)
            QApplication.setDesktopFileName("rag-assistant")

        visual = _load_visual()
        app = QApplication(sys.argv)
        icon = QIcon(_icon_path)
        app.setWindowIcon(icon)
        app.setApplicationName("RAG Assistant")
        app.setStyleSheet(_stylesheet(visual.get("theme", "dark"), visual.get("font_size", 15)))
        window = MainWindow(config, visual)
        window.setWindowIcon(icon)
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
