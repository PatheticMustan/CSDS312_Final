#!/bin/bash
#SBATCH --job-name=pd-bench
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$SLURM_SUBMIT_DIR}
RAW_DIR=${RAW_DIR:-$ROOT_DIR/data/raw/amp-parkinsons-disease-progression-prediction}
DATA=${DATA:-$ROOT_DIR/data/cleaned/final_dataset.csv}
SPLIT=${SPLIT:-patient}
OUTPUT=${OUTPUT:-$ROOT_DIR/results/benchmarks/${SPLIT}_benchmark_${SLURM_JOB_ID}.csv}
N_JOBS_VALUES=${N_JOBS_VALUES:-1,2,4,8}
MODELS=${MODELS:-random_forest,xgboost}
TARGETS=${TARGETS:-updrs_1}
REPEAT=${REPEAT:-3}
SEED=${SEED:-42}
MODEL_PARAMS=${MODEL_PARAMS:-{}}
KEEP_RUNS=${KEEP_RUNS:-0}
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
  python scripts/benchmark_pipeline.py
  --raw-dir "$RAW_DIR"
  --data "$DATA"
  --output "$OUTPUT"
  --n-jobs-values "$N_JOBS_VALUES"
  --models "$MODELS"
  --targets "$TARGETS"
  --split "$SPLIT"
  --repeat "$REPEAT"
  --seed "$SEED"
  --model-params "$MODEL_PARAMS"
)

if [ "$KEEP_RUNS" = "1" ]; then
  cmd+=(--keep-runs)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
