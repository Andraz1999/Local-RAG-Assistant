@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  RAG Assistant - Windows Launcher
::  Run this file to set up and start the application.
::  First run will download models and install packages.
:: ============================================================

title RAG Assistant Launcher
color 0A

echo.
echo  =====================================================
echo   RAG Assistant - Launcher
echo  =====================================================
echo.

:: ── Working directory: always the folder where this .bat lives ──
cd /d "%~dp0"


:: ════════════════════════════════════════════════════════════
::  STEP 1 — Check Python
:: ════════════════════════════════════════════════════════════

echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python was not found.
    echo.
    echo  Please install Python 3.10 or newer from:
    echo    https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: During installation, check the box that says
    echo  "Add Python to PATH" before clicking Install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  [OK] Found Python %PY_VER%


:: ════════════════════════════════════════════════════════════
::  STEP 2 — Check Ollama
:: ════════════════════════════════════════════════════════════

echo.
echo [2/5] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Ollama was not found.
    echo.
    echo  Please install Ollama from:
    echo    https://ollama.com/download
    echo.
    echo  After installing, re-run this script.
    echo.
    pause
    exit /b 1
)
echo  [OK] Ollama is installed.

:: Make sure the Ollama service is running
echo  Starting Ollama service (if not already running)...
start /b "" ollama serve >nul 2>&1
timeout /t 3 /nobreak >nul


:: ════════════════════════════════════════════════════════════
::  STEP 3 — Python virtual environment
:: ════════════════════════════════════════════════════════════

echo.
echo [3/5] Setting up Python environment...

if not exist "project\.venv" (
    echo  Creating virtual environment for the first time...
    python -m venv "project\.venv"
    if errorlevel 1 (
        echo.
        echo  [ERROR] Failed to create virtual environment.
        echo  Make sure the 'venv' module is available in your Python installation.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
) else (
    echo  [OK] Virtual environment already exists.
)

:: Activate
call "project\.venv\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo  [ERROR] Could not activate virtual environment.
    pause
    exit /b 1
)

:: Install / update packages
echo.
echo  Installing Python packages (this may take several minutes on first run)...
echo  Packages: PyQt6, torch, faiss, sentence-transformers, unstructured, and more.
echo.
pip install -r "project\requirements.txt" --quiet --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo  [ERROR] Package installation failed.
    echo  Check your internet connection and try again.
    echo  If the problem persists, run this manually for details:
    echo    project\.venv\Scripts\pip install -r project\requirements.txt
    pause
    exit /b 1
)
echo  [OK] All Python packages are installed.


:: ════════════════════════════════════════════════════════════
::  STEP 4 — Pull Ollama models
:: ════════════════════════════════════════════════════════════

echo.
echo [4/5] Checking Ollama models...
echo  (Models are several GB each and only download once.)
echo.

:: Use the external helper script to avoid inline Python quoting issues
for /f "delims=" %%m in ('python get_models.py "project\config.json"') do (
    echo  Pulling model: %%m
    ollama pull %%m
    if errorlevel 1 (
        echo.
        echo  [WARNING] Could not pull model: %%m
        echo  The app may not work correctly without it.
        echo  You can pull it manually later with:  ollama pull %%m
        echo.
    ) else (
        echo  [OK] %%m is ready.
    )
)

echo.
echo  [OK] All models are ready.


:: ════════════════════════════════════════════════════════════
::  STEP 5 — Launch the application
:: ════════════════════════════════════════════════════════════

echo.
echo [5/5] Launching RAG Assistant...
echo.
echo  =====================================================
echo   The application is starting.
echo  =====================================================
echo.

cd project
python main.py
if errorlevel 1 (
    echo.
    echo  [ERROR] The application exited with an error.
    echo  See the output above for details.
    echo.
    pause
)

endlocal
