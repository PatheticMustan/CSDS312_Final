#!/bin/bash
#SBATCH --job-name=pd-importance
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$SLURM_SUBMIT_DIR}
RUN_DIR=${RUN_DIR:-}
OUTPUT=${OUTPUT:-}
TOP_K=${TOP_K:-20}
VENV_DIR=${VENV_DIR:-$ROOT_DIR/.venv-hpc}
PYTHON_MODULE=${PYTHON_MODULE:-Python/3.11.5-GCCcore-13.2.0}

if [ -z "$RUN_DIR" ]; then
  echo "RUN_DIR must point to a completed results/runs/... directory" >&2
  exit 1
fi

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
  source /etc/profile.d/lmod.sh 2>/dev/null || true
fi

module purge
module load "$PYTHON_MODULE"

cd "$ROOT_DIR"

if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/bin/activate"
fi

cmd=(
  python scripts/export_feature_importance.py
  --run-dir "$RUN_DIR"
  --top-k "$TOP_K"
)

if [ -n "$OUTPUT" ]; then
  cmd+=(--output "$OUTPUT")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
