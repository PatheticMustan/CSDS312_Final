#!/bin/bash
#SBATCH --job-name=pd-build
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$SLURM_SUBMIT_DIR}
RAW_DIR=${RAW_DIR:-$ROOT_DIR/data/raw/amp-parkinsons-disease-progression-prediction}
OUTPUT=${OUTPUT:-$ROOT_DIR/data/cleaned/final_dataset.csv}
N_JOBS=${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}
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

python scripts/build_training_data.py \
  --raw-dir "$RAW_DIR" \
  --output "$OUTPUT" \
  --n-jobs "$N_JOBS"
