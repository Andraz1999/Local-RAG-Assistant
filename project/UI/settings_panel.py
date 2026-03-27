"""
UI/settings_panel.py
--------------------
Full-screen settings dialog with three tabs:
  1. Main      — folders, retrieval mode, sparse model, k
  2. Visual    — font size slider, dark/light theme
  3. Advanced  — models (comboboxes), chunking params, rrf_constant

Changes vs original:
  - Ollama Base URL field removed from Advanced (it is a fixed internal detail).
  - "Confirm Advanced Settings" rebuild button removed.
  - OK button flow:
      • If embedding model or SPLADE model changed → show rebuild warning dialog
        with Confirm / Cancel. Confirm saves config + emits rebuild_requested.
        Cancel returns to Settings.
      • Otherwise → save normally via settings_applied.
  - Warning banner updated: only embedding/SPLADE cause a rebuild.
  - All QComboBox widgets replaced with WheelIgnoreComboBox so that
    scrolling the mouse wheel doesn't change them.

No src imports. Emits signals; main.py does all the real work.
"""

from __future__ import annotations

import json

from PyQt6.QtCore import pyqtSignal, Qt, QEvent
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QSlider, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget, QMessageBox,
)

DEFAULT_VISUAL = {"font_size": 10, "theme": "dark"}


# ---------------------------------------------------------------------------
# Mouse-wheel-safe combo box
# ---------------------------------------------------------------------------

class WheelIgnoreComboBox(QComboBox):
    """A QComboBox that ignores mouse-wheel events (passes them up to parent)."""

    def wheelEvent(self, event: QEvent) -> None:  # type: ignore[override]
        event.ignore()   # let the scroll area handle it


# ---------------------------------------------------------------------------
# Rebuild-warning dialog
# ---------------------------------------------------------------------------

class _RebuildWarningDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rebuild Required")
        self.setModal(True)
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(14)

        icon_row = QHBoxLayout()
        warn_icon = QLabel("⚠")
        warn_icon.setFont(QFont("Segoe UI", 24))
        warn_icon.setStyleSheet("color: #fb923c;")
        title = QLabel("Embedding or SPLADE model changed")
        title.setFont(QFont("Segoe UI Semibold", 12))
        icon_row.addWidget(warn_icon)
        icon_row.addWidget(title, 1)
        layout.addLayout(icon_row)

        msg = QLabel(
            "You have changed the <b>Embedding model</b> or the <b>SPLADE model</b>.<br><br>"
            "These models determine how your documents are encoded into the vector database. "
            "Changing them requires the database to be completely rebuilt, which may take "
            "a while depending on the number of documents.<br><br>"
            "Click <b>Confirm</b> to save your settings and start rebuilding the database, "
            "or <b>Cancel</b> to go back to settings without saving."
        )
        msg.setWordWrap(True)
        msg.setObjectName("FieldLabel")
        layout.addWidget(msg)

        btn_row = QHBoxLayout()
        self._cancel_btn = QPushButton("Cancel")
        self._confirm_btn = QPushButton("Confirm — Rebuild DB")
        self._confirm_btn.setObjectName("RebuildBtn")
        self._cancel_btn.clicked.connect(self.reject)
        self._confirm_btn.clicked.connect(self.accept)
        btn_row.addStretch()
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._confirm_btn)
        layout.addLayout(btn_row)


# ---------------------------------------------------------------------------
# Main settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    settings_applied  = pyqtSignal(dict, dict)
    rebuild_requested = pyqtSignal(dict, dict)
    reset_requested   = pyqtSignal(dict, dict)   # manual "Reset DB" button
    index_requested   = pyqtSignal()

    def __init__(self, config: dict, visual: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(720, 672)
        self.resize(820, 768)
        self.setModal(True)
        self.setObjectName("SettingsDialog")

        self._config = json.loads(json.dumps(config))
        self._visual = json.loads(json.dumps(visual))

        # Remember original embedding/splade model indices to detect changes
        self._orig_emb_idx    = self._config.get("embedding", {}).get("current_embedding_model", 0)
        self._orig_splade_idx = self._config.get("embedding", {}).get("current_splade_model", 0)

        self._setup_ui()
        self._load_config(self._config)
        self._load_visual(self._visual)

    # ── UI ──────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(16)

        title = QLabel("Settings")
        title.setObjectName("DialogTitle")
        title.setFont(QFont("Segoe UI Semibold", 16))
        root.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.setObjectName("SettingsTabs")
        self._tabs.addTab(self._build_main_tab(),     "  Main  ")
        self._tabs.addTab(self._build_visual_tab(),   "  Visual  ")
        self._tabs.addTab(self._build_advanced_tab(), "  Advanced  ")
        root.addWidget(self._tabs, 1)

        row = QHBoxLayout()
        scan_btn = QPushButton("🔍  Scan the database")
        scan_btn.setObjectName("ScanBtn")
        scan_btn.clicked.connect(self._on_scan)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)

        row.addWidget(scan_btn)
        row.addStretch()
        row.addWidget(btn_box)
        root.addLayout(row)

    # ── Tabs ────────────────────────────────────────────────────────────

    def _build_main_tab(self) -> QWidget:
        tab, layout = self._scrollable_tab()

        # ── Database ─────────────────────────────
        folders_box, fl = self._group("Database")

        fl.addWidget(self._lbl("Database with the PDFs"))

        r1 = QHBoxLayout()
        self._input_folder = QLineEdit()
        self._input_folder.setPlaceholderText("Path to folder containing PDFs…")

        b1 = self._small_btn("Browse")
        b1.clicked.connect(self._browse_input)

        r1.addWidget(self._input_folder)
        r1.addWidget(b1)
        fl.addLayout(r1)

        layout.addWidget(folders_box)

        # ── Retrieval ────────────────────────────
        ret_box, rl = self._group("Retrieval")

        rl.addWidget(self._label_with_help(
            "Retrieval Mode",
            "Meaning: Searches for chunks with similar meaning as the question.\n"
            "Words: Searches for chunks with the same words.\n"
            "Hybrid: Combination of both."
        ))

        self._mode_combo = WheelIgnoreComboBox()
        self._mode_combo.addItem("Meaning", "dense")
        self._mode_combo.addItem("Hybrid", "hybrid")
        self._mode_combo.addItem("Words", "sparse")
        rl.addWidget(self._mode_combo)

        rl.addWidget(self._label_with_help(
            "Sparse Model",
            "Only applies if Retrieval Mode is Words or Hybrid.\n\n"
            "Powerful but slow:\nUses SPLADE to find semantically similar words.\n\n"
            "Less powerful but fast:\nUses BM25 for exact keyword matching."
        ))

        self._sparse_combo = WheelIgnoreComboBox()
        self._sparse_combo.addItem("Powerful but slow", "splade")
        self._sparse_combo.addItem("Less powerful but fast", "bm25")
        rl.addWidget(self._sparse_combo)

        kr = QHBoxLayout()
        kr.addWidget(self._label_with_help(
            "K (nearest neighbours)",
            "How many of the most relevant text chunks are retrieved\n"
            "and used to answer your question."
        ), 1)

        self._k_spin = QSpinBox()
        self._k_spin.setRange(1, 100)
        self._k_spin.setFixedWidth(80)

        kr.addWidget(self._k_spin)
        rl.addLayout(kr)

        layout.addWidget(ret_box)
        layout.addStretch()
        return tab

    def _build_visual_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 20, 8, 12)
        layout.setSpacing(28)

        theme_box, tl = self._group("Theme")
        tl.addWidget(self._lbl("Choose between a dark or light interface."))

        tr = QHBoxLayout()
        self._dark_btn  = QPushButton("🌙   Dark Mode")
        self._light_btn = QPushButton("☀   Light Mode")

        for btn in (self._dark_btn, self._light_btn):
            btn.setCheckable(True)
            btn.setFixedHeight(52)
            btn.setObjectName("ThemeChoiceBtn")

        self._dark_btn.clicked.connect(lambda: self._select_theme("dark"))
        self._light_btn.clicked.connect(lambda: self._select_theme("light"))

        tr.addWidget(self._dark_btn)
        tr.addWidget(self._light_btn)
        tl.addLayout(tr)
        layout.addWidget(theme_box)

        font_box, fnl = self._group("Interface Font Size")
        self._font_preview = QLabel("The quick brown fox jumped over the lazy dog.")
        self._font_preview.setObjectName("FontPreview")
        fnl.addWidget(self._font_preview)

        sr = QHBoxLayout()
        self._font_slider = QSlider(Qt.Orientation.Horizontal)
        self._font_slider.setRange(8, 18)
        self._font_slider.valueChanged.connect(self._on_font_slider)
        self._font_size_lbl = QLabel("10pt")
        self._font_size_lbl.setObjectName("FontSizeLabel")

        sr.addWidget(self._font_slider)
        sr.addWidget(self._font_size_lbl)
        fnl.addLayout(sr)

        layout.addWidget(font_box)
        layout.addStretch()
        return tab

    def _build_advanced_tab(self) -> QWidget:
        tab, layout = self._scrollable_tab()

        # Informational banner (not a warning to avoid building)
        warn = QLabel(
            "ℹ  Only the <b>Embedding Model</b> and <b>SPLADE Model</b> require a full "
            "database rebuild when changed. All other advanced settings are applied directly."
        )
        warn.setObjectName("WarnBannerEmbedding")
        warn.setWordWrap(True)
        layout.addWidget(warn)

        # ── Models ───────────────────────────────
        models_box, ml = self._group("Models")

        ml.addWidget(self._label_with_help(
            "Rewriter Model  (set to 'Disabled' to turn off)",
            "Rewrites your question before searching to improve retrieval quality.\n"
            "Uses a local Ollama model. Set to Disabled to skip rewriting."
        ))
        self._rewriter_combo = WheelIgnoreComboBox()
        ml.addWidget(self._rewriter_combo)

        ml.addWidget(self._label_with_help(
            "Reasoner Model",
            "The model that reads the retrieved chunks and generates the final answer.\n"
            "Uses a local Ollama model."
        ))
        self._reasoner_combo = WheelIgnoreComboBox()
        ml.addWidget(self._reasoner_combo)

        ml.addWidget(self._label_with_help(
            "Embedding Model  ⚠ rebuild if changed",
            "The model that converts text into vectors for semantic search.\n"
            "Ollama models run locally via the Ollama server.\n"
            "HuggingFace models are downloaded and run locally.\n"
            "The provider (Ollama / HuggingFace) is set automatically.\n\n"
            "Changing this model requires rebuilding the vector database."
        ))
        self._emb_combo = WheelIgnoreComboBox()
        ml.addWidget(self._emb_combo)

        ml.addWidget(self._label_with_help(
            "SPLADE Model  ⚠ rebuild if changed",
            "The sparse encoder used when Retrieval Mode is Words or Hybrid.\n"
            "Only applies when sparse_model is set to 'splade'.\n\n"
            "Changing this model requires rebuilding the vector database."
        ))
        self._splade_combo = WheelIgnoreComboBox()
        ml.addWidget(self._splade_combo)

        layout.addWidget(models_box)

        # ── Chunking ─────────────────────────────
        chunk_box, cl = self._group("Chunking")

        for label_txt, attr, lo, hi, step, tooltip in [
            (
                "Max Chunk Characters",
                "_max_chars", 100, 10000, 100,
                "Maximum number of characters allowed in a single chunk.\n"
                "Larger chunks carry more context but may dilute relevance."
            ),
            (
                "New After N Chars",
                "_new_after", 100, 10000, 100,
                "Start a new chunk after this many characters, even if a\n"
                "natural boundary hasn't been reached yet."
            ),
            (
                "Chunk Overlap",
                "_overlap", 0, 2000, 50,
                "Number of characters repeated between consecutive chunks.\n"
                "Overlap helps avoid cutting sentences at chunk boundaries."
            ),
        ]:
            row = QHBoxLayout()
            row.addWidget(self._label_with_help(label_txt, tooltip), 1)
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setFixedWidth(100)
            setattr(self, attr, spin)
            row.addWidget(spin)
            cl.addLayout(row)

        layout.addWidget(chunk_box)

        # ── Retrieval advanced ────────────────────
        ret_adv_box, ral = self._group("Retrieval (Advanced)")

        rrf_row = QHBoxLayout()
        rrf_row.addWidget(self._label_with_help(
            "RRF Constant",
            "Reciprocal Rank Fusion constant (k).\n"
            "Used when combining dense and sparse results in Hybrid mode.\n"
            "Higher values reduce the impact of top-ranked results.\n"
            "Default is 60."
        ), 1)
        self._rrf_spin = QSpinBox()
        self._rrf_spin.setRange(1, 1000)
        self._rrf_spin.setFixedWidth(100)
        rrf_row.addWidget(self._rrf_spin)
        ral.addLayout(rrf_row)

        layout.addWidget(ret_adv_box)

        # ── Reset Database ────────────────────────
        reset_box, resetl = self._group("Reset Database")

        reset_desc = QLabel(
            "Wipe the entire vector database and rebuild it from scratch. "
            "All current settings will be saved and applied before rebuilding."
        )
        reset_desc.setObjectName("FieldLabel")
        reset_desc.setWordWrap(True)
        resetl.addWidget(reset_desc)

        reset_btn_row = QHBoxLayout()
        reset_btn_row.addStretch()
        reset_db_btn = QPushButton("🗑  Reset Database")
        reset_db_btn.setObjectName("ResetDbBtn")
        reset_db_btn.clicked.connect(self._on_reset_db)
        reset_btn_row.addWidget(reset_db_btn)
        resetl.addLayout(reset_btn_row)

        layout.addWidget(reset_box)
        layout.addStretch()
        return tab

    # ── Helpers ─────────────────────────────────────────────────────────

    def _scrollable_tab(self):
        tab = QWidget()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)

        layout = QVBoxLayout(container)

        if not hasattr(self, "_refs"):
            self._refs = []
        self._refs.append((scroll, container))

        return tab, layout

    def _group(self, title: str):
        box = QGroupBox(title); box.setObjectName("SettingsGroup")
        inner = QVBoxLayout(box); inner.setSpacing(8); inner.setContentsMargins(14, 12, 14, 12)
        return box, inner

    def _lbl(self, text: str) -> QLabel:
        l = QLabel(text); l.setObjectName("FieldLabel"); return l

    def _lbl_s(self, text: str) -> QLabel:
        l = QLabel(text); l.setObjectName("SliderLabel"); return l

    def _small_btn(self, text: str) -> QPushButton:
        b = QPushButton(text); b.setObjectName("SmallBtn"); b.setFixedWidth(80); return b

    def _label_with_help(self, text: str, tooltip: str) -> QWidget:
        row = QHBoxLayout()
        row.setSpacing(6)

        label = QLabel(text)
        label.setObjectName("FieldLabel")

        help_icon = QLabel("(?)")
        help_icon.setObjectName("HelpIcon")
        help_icon.setCursor(Qt.CursorShape.PointingHandCursor)

        row.addWidget(label)
        row.addWidget(help_icon)
        row.addStretch()

        container = QWidget()
        container.setLayout(row)
        container.setToolTip(tooltip)
        help_icon.setToolTip(tooltip)
        return container

    # ── Populate combo helpers ───────────────────────────────────────────

    def _populate_embedding_combo(self, config: dict) -> None:
        self._emb_combo.clear()
        emb = config.get("embedding", {})
        for name in emb.get("ollama_embedding_model_names", []):
            self._emb_combo.addItem(f"[Ollama]  {name}", ("ollama", name))
        for name in emb.get("hf_embedding_model_names", []):
            self._emb_combo.addItem(f"[HF]  {name}", ("huggingface", name))

        idx = emb.get("current_embedding_model", 0)
        if 0 <= idx < self._emb_combo.count():
            self._emb_combo.setCurrentIndex(idx)

    def _populate_splade_combo(self, config: dict) -> None:
        self._splade_combo.clear()
        emb = config.get("embedding", {})
        for name in emb.get("splade_model_names", []):
            self._splade_combo.addItem(name)
        idx = emb.get("current_splade_model", 0)
        if 0 <= idx < self._splade_combo.count():
            self._splade_combo.setCurrentIndex(idx)

    def _populate_rewriter_combo(self, config: dict) -> None:
        self._rewriter_combo.clear()
        for name in config.get("rewriter", {}).get("model_names", []):
            self._rewriter_combo.addItem(name)
        idx = config.get("rewriter", {}).get("current_rewriter_model", 0)
        if 0 <= idx < self._rewriter_combo.count():
            self._rewriter_combo.setCurrentIndex(idx)

    def _populate_reasoner_combo(self, config: dict) -> None:
        self._reasoner_combo.clear()
        for name in config.get("reasoner", {}).get("model_names", []):
            self._reasoner_combo.addItem(name)
        idx = config.get("reasoner", {}).get("current_reasoner_model", 0)
        if 0 <= idx < self._reasoner_combo.count():
            self._reasoner_combo.setCurrentIndex(idx)

    # ── Load / collect ───────────────────────────────────────────────────

    def _load_config(self, c: dict) -> None:
        self._input_folder.setText(c.get("paths", {}).get("input_folder", ""))

        r = c.get("retrieval", {})
        mode = r.get("mode", "hybrid")
        for i in range(self._mode_combo.count()):
            if self._mode_combo.itemData(i) == mode:
                self._mode_combo.setCurrentIndex(i)
                break

        sparse = r.get("sparse_model", "splade")
        for i in range(self._sparse_combo.count()):
            if self._sparse_combo.itemData(i) == sparse:
                self._sparse_combo.setCurrentIndex(i)
                break

        self._k_spin.setValue(r.get("k", 10))
        self._rrf_spin.setValue(r.get("rrf_constant", 60))

        self._populate_embedding_combo(c)
        self._populate_splade_combo(c)
        self._populate_rewriter_combo(c)
        self._populate_reasoner_combo(c)

        ch = c.get("chunking", {})
        self._max_chars.setValue(ch.get("max_characters",    2000))
        self._new_after.setValue(ch.get("new_after_n_chars", 1500))
        self._overlap.setValue(  ch.get("overlap",           500))

    def _load_visual(self, v: dict) -> None:
        size = v.get("font_size", 10)
        self._font_slider.setValue(size)
        self._font_size_lbl.setText(f"{size}pt")
        self._font_preview.setFont(QFont("Segoe UI", size))
        self._select_theme(v.get("theme", "dark"), save=False)

    def _collect_config(self) -> dict:
        emb_data    = self._emb_combo.currentData()
        provider, _ = emb_data if emb_data else ("huggingface", "")

        return {
            "paths": {
                "input_folder": self._input_folder.text(),
                "db_folder":    self._config.get("paths", {}).get("db_folder", "../vector_db"),
            },
            "retrieval": {
                "mode":         self._mode_combo.currentData(),
                "sparse_model": self._sparse_combo.currentData(),
                "k":            self._k_spin.value(),
                "rrf_constant": self._rrf_spin.value(),
            },
            "chunking": {
                "strategy":           self._config.get("chunking", {}).get("strategy", "by_title"),
                "max_characters":     self._max_chars.value(),
                "new_after_n_chars":  self._new_after.value(),
                "overlap":            self._overlap.value(),
            },
            "partition": self._config.get("partition", {}),
            "embedding": {
                # Ollama base URL is an internal fixed setting — preserved as-is
                "ollama_base_url":              self._config.get("embedding", {}).get("ollama_base_url", ""),
                "ollama_embedding_model_names": self._config["embedding"].get("ollama_embedding_model_names", []),
                "hf_embedding_model_names":     self._config["embedding"].get("hf_embedding_model_names", []),
                "current_embedding_model":      self._emb_combo.currentIndex(),
                "splade_model_names":           self._config["embedding"].get("splade_model_names", []),
                "current_splade_model":         self._splade_combo.currentIndex(),
                "batch_size":                   self._config["embedding"].get("batch_size", 32),
                "provider":                     provider,
            },
            "rewriter": {
                "model_names":            self._config.get("rewriter", {}).get("model_names", []),
                "current_rewriter_model": self._rewriter_combo.currentIndex(),
            },
            "reasoner": {
                "model_names":            self._config.get("reasoner", {}).get("model_names", []),
                "current_reasoner_model": self._reasoner_combo.currentIndex(),
            },
        }

    def _collect_visual(self) -> dict:
        return {
            "font_size": self._font_slider.value(),
            "theme":     self._visual.get("theme", "dark"),
        }

    def _embedding_changed(self) -> bool:
        return self._emb_combo.currentIndex() != self._orig_emb_idx

    def _splade_changed(self) -> bool:
        return self._splade_combo.currentIndex() != self._orig_splade_idx

    # ── Slots ────────────────────────────────────────────────────────────

    def _browse_input(self) -> None:
        f = QFileDialog.getExistingDirectory(self)
        if f:
            self._input_folder.setText(f)

    def _on_font_slider(self, value: int) -> None:
        self._font_size_lbl.setText(f"{value}pt")
        self._font_preview.setFont(QFont("Segoe UI", value))

    def _select_theme(self, theme: str, save: bool = True) -> None:
        self._dark_btn.setChecked(theme == "dark")
        self._light_btn.setChecked(theme == "light")
        if save:
            self._visual["theme"] = theme

    def _on_reset_db(self) -> None:
        """Show confirmation dialog; if confirmed, save all settings and emit reset_requested."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Reset Database")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText("<b>You are about to reset the entire vector database.</b>")
        msg.setInformativeText(
            "This will wipe all indexed data and rebuild the database from scratch "
            "using the current settings. Depending on the number of documents, "
            "this may take a while.\n\nDo you want to continue?"
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
        msg.button(QMessageBox.StandardButton.Yes).setText("Confirm — Reset && Rebuild")
        if msg.exec() == QMessageBox.StandardButton.Yes:
            config = self._collect_config()
            visual = self._collect_visual()
            self.reset_requested.emit(config, visual)
            self.accept()

    def _on_scan(self) -> None:
        """Close the settings dialog first, then trigger indexing."""
        config = self._collect_config()
        visual = self._collect_visual()
        self.settings_applied.emit(config, visual)
        self.accept()
        self.index_requested.emit()

    def _on_ok(self) -> None:
        config = self._collect_config()
        visual = self._collect_visual()

        if self._embedding_changed() or self._splade_changed():
            # Show rebuild warning dialog
            warn_dlg = _RebuildWarningDialog(self)
            if warn_dlg.exec() == QDialog.DialogCode.Accepted:
                # User confirmed — emit rebuild signal and close
                self.rebuild_requested.emit(config, visual)
                self.accept()
            # else: user cancelled — stay in settings (do nothing)
        else:
            # Normal save
            self.settings_applied.emit(config, visual)
            self.accept()