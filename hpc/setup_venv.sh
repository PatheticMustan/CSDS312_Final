#!/bin/bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_MODULE=${PYTHON_MODULE:-Python/3.11.5-GCCcore-13.2.0}
PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-$ROOT_DIR/.venv-hpc}

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
  source /etc/profile.d/lmod.sh 2>/dev/null || true
fi

module purge
module load "$PYTHON_MODULE"

cd "$ROOT_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-hpc.txt

echo "Created HPC-ready virtual environment at: $VENV_DIR"
