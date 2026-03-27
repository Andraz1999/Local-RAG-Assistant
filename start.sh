#!/usr/bin/env bash
# ============================================================
#  RAG Assistant - Linux / macOS Launcher
#  Run this file to set up and start the application.
#  First run will download models and install packages.
#
#  Usage:
#    chmod +x start.sh   (once, to make it executable)
#    ./start.sh
# ============================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${RESET} $*"; }
warn() { echo -e "  ${YELLOW}[WARNING]${RESET} $*"; }
err()  { echo -e "  ${RED}[ERROR]${RESET} $*"; }
info() { echo -e "  ${CYAN}$*${RESET}"; }

# ── Working directory: always the folder where this script lives ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_DIR="$SCRIPT_DIR/project"

echo
echo -e "${BOLD} =====================================================${RESET}"
echo -e "${BOLD}  RAG Assistant - Launcher${RESET}"
echo -e "${BOLD} =====================================================${RESET}"
echo


# ════════════════════════════════════════════════════════════
#  STEP 1 — Check Python
# ════════════════════════════════════════════════════════════

echo -e "${BOLD}[1/5] Checking Python...${RESET}"

if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    err "Python was not found."
    echo
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  On macOS, the easiest way is:"
        echo "    brew install python"
        echo "  Or download from: https://www.python.org/downloads/"
    else
        echo "  On Ubuntu/Debian:  sudo apt install python3 python3-venv python3-pip"
        echo "  On Fedora:         sudo dnf install python3"
        echo "  Or download from:  https://www.python.org/downloads/"
    fi
    echo
    exit 1
fi

PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 10 ]]; }; then
    err "Python $PY_VER is too old. Version 3.10 or newer is required."
    echo "  Please install a newer version from: https://www.python.org/downloads/"
    exit 1
fi

ok "Found Python $PY_VER  ($PYTHON)"


# ════════════════════════════════════════════════════════════
#  STEP 2 — Check Ollama
# ════════════════════════════════════════════════════════════

echo
echo -e "${BOLD}[2/5] Checking Ollama...${RESET}"

if ! command -v ollama &>/dev/null; then
    err "Ollama was not found."
    echo
    echo "  Please install Ollama from: https://ollama.com/download"
    echo
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  On macOS you can also use:  brew install ollama"
    else
        echo "  On Linux, the quick install is:"
        echo "    curl -fsSL https://ollama.com/install.sh | sh"
    fi
    echo
    exit 1
fi

ok "Ollama is installed."

# Start Ollama service in the background if it isn't already running
if ! curl -sf http://localhost:11434 &>/dev/null; then
    info "Starting Ollama service..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open -a Ollama &>/dev/null || ollama serve &>/dev/null &
    else
        ollama serve &>/dev/null &
    fi
    for i in {1..10}; do
        sleep 1
        if curl -sf http://localhost:11434 &>/dev/null; then break; fi
        if [[ $i -eq 10 ]]; then
            warn "Ollama service did not start in time. Models may fail to pull."
        fi
    done
fi

ok "Ollama service is running."


# ════════════════════════════════════════════════════════════
#  STEP 3 — Python virtual environment
# ════════════════════════════════════════════════════════════

echo
echo -e "${BOLD}[3/5] Setting up Python environment...${RESET}"

if ! $PYTHON -m venv --help &>/dev/null; then
    err "The Python 'venv' module is not available."
    echo
    echo "  On Ubuntu/Debian, install it with:"
    echo "    sudo apt install python3-venv"
    echo
    exit 1
fi

if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
    info "Creating virtual environment for the first time..."
    $PYTHON -m venv "$PROJECT_DIR/.venv"
    ok "Virtual environment created."
else
    ok "Virtual environment already exists."
fi

# Activate
# shellcheck disable=SC1091
source "$PROJECT_DIR/.venv/bin/activate"

pip install --upgrade pip --quiet --disable-pip-version-check

echo
info "Installing Python packages (this may take several minutes on first run)..."
info "Packages: PyQt6, torch, faiss, sentence-transformers, unstructured, and more."
echo

pip install -r "$PROJECT_DIR/requirements.txt" --quiet --disable-pip-version-check
if [[ $? -ne 0 ]]; then
    err "Package installation failed."
    echo "  Check your internet connection and try again."
    echo "  For detailed output, run:  $PROJECT_DIR/.venv/bin/pip install -r $PROJECT_DIR/requirements.txt"
    exit 1
fi

ok "All Python packages are installed."


# ════════════════════════════════════════════════════════════
#  STEP 4 — Pull Ollama models (read from config.json)
# ════════════════════════════════════════════════════════════

echo
echo -e "${BOLD}[4/5] Checking Ollama models...${RESET}"
info "(Models are several GB each and only download once.)"
echo

MODELS=$(python - <<EOF
import json, sys
try:
    c = json.load(open("$PROJECT_DIR/config.json", encoding="utf-8"))
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
EOF
)

if [[ -z "$MODELS" ]]; then
    warn "No models found in config.json. Skipping model downloads."
else
    while IFS= read -r model; do
        [[ -z "$model" ]] && continue
        info "Pulling model: $model"
        if ollama pull "$model"; then
            ok "$model is ready."
        else
            warn "Could not pull model: $model"
            echo "    You can pull it manually later with:  ollama pull $model"
        fi
        echo
    done <<< "$MODELS"
fi

ok "All models are ready."


# ════════════════════════════════════════════════════════════
#  STEP 4.5 — Install .desktop file (Linux only)
# ════════════════════════════════════════════════════════════

if [[ "$OSTYPE" != "darwin"* ]]; then
    DESKTOP_DIR="$HOME/.local/share/applications"
    DESKTOP_FILE="$DESKTOP_DIR/rag-assistant.desktop"
    mkdir -p "$DESKTOP_DIR"

    cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Name=RAG Assistant
Exec=$PROJECT_DIR/.venv/bin/python $PROJECT_DIR/main.py
Icon=$PROJECT_DIR/icon.png
Type=Application
Categories=Utility;
StartupWMClass=RAG Assistant
EOF

    # Tell the DE to re-scan so the icon shows immediately
    update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
    ok ".desktop file installed → $DESKTOP_FILE"
fi

# ════════════════════════════════════════════════════════════
#  STEP 5 — Launch the application
# ════════════════════════════════════════════════════════════

echo
echo -e "${BOLD}[5/5] Launching RAG Assistant...${RESET}"
echo
echo -e "${BOLD} =====================================================${RESET}"
echo -e "${BOLD}  The application is starting.${RESET}"
echo -e "${BOLD} =====================================================${RESET}"
echo

# On Linux, PyQt6 sometimes needs a display server hint
if [[ "$OSTYPE" != "darwin"* ]]; then
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
fi

cd "$PROJECT_DIR"
python main.py
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo
    err "The application exited with error code $EXIT_CODE."
    echo "  See the output above for details."
    exit $EXIT_CODE
fi