#!/bin/bash
#SBATCH --job-name=pd-train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$SLURM_SUBMIT_DIR}
DATA=${DATA:-$ROOT_DIR/data/cleaned/final_dataset.csv}
MODEL=${MODEL:-random_forest}
TARGET=${TARGET:-updrs_1}
SPLIT=${SPLIT:-patient}
SEED=${SEED:-42}
N_JOBS=${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/results}
MODEL_PARAMS=${MODEL_PARAMS:-}
VENV_DIR=${VENV_DIR:-$ROOT_DIR/.venv-hpc}
PYTHON_MODULE=${PYTHON_MODULE:-Python/3.11.5-GCCcore-13.2.0}

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
  python scripts/train_baseline.py
  --data "$DATA"
  --model "$MODEL"
  --target "$TARGET"
  --split "$SPLIT"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
  --n-jobs "$N_JOBS"
)

if [ -n "$MODEL_PARAMS" ]; then
  cmd+=(--model-params "$MODEL_PARAMS")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
