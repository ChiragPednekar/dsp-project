#!/usr/bin/env bash
# Launch the equalizer GUI. Creates the virtualenv and installs dependencies
# on first run, then starts Streamlit and opens the app in your browser.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

if [ ! -d "$VENV" ]; then
  echo "==> Creating virtualenv ($($PYTHON --version))"
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  echo "==> Installing dependencies (this takes a minute the first time)"
  "$VENV/bin/pip" install --quiet -r requirements.txt
fi

# The demo clip is generated rather than committed as a binary blob.
if [ ! -f samples/demo.wav ]; then
  echo "==> Generating the demo clip"
  "$VENV/bin/python" tools/make_sample.py
fi

echo "==> Starting the equalizer at http://localhost:8501"
exec "$VENV/bin/python" -m streamlit run app.py
