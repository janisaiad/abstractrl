#!/usr/bin/env bash
# Nettoie les caches d’installation (pip / uv) sans supprimer les venvs.
# Peut libérer plusieurs Go après des installs PyTorch répétées.
set -euo pipefail
if command -v pip >/dev/null 2>&1; then
  pip cache purge 2>/dev/null || true
fi
if command -v uv >/dev/null 2>&1; then
  uv cache prune 2>/dev/null || true
fi
echo "Caches pip/uv purgés (les .venv ne sont pas touchés)."
