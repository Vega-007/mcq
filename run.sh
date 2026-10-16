#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Intelligent MCQ Generator Streamlit App..."

if [ -f ".venv/bin/streamlit" ]; then
    exec .venv/bin/streamlit run app.py "$@"
elif command -v streamlit >/dev/null 2>&1; then
    exec streamlit run app.py "$@"
else
    exec python3 -m streamlit run app.py "$@"
fi
