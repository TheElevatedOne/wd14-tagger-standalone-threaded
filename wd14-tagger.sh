#!/usr/bin/bash
SCRIPT_DIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")

exec "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/main.py" "$@"
