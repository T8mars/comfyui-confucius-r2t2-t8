#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if command -v python3.12 >/dev/null 2>&1; then
    exec python3.12 "$SCRIPT_DIR/setup_worker.py" "$@"
fi
exec python3 "$SCRIPT_DIR/setup_worker.py" "$@"
