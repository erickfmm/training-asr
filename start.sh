#!/usr/bin/env bash
# ============================================================================
# start.sh — Lanza el fine-tuning ASR en español sobre Auden TTA m10.
# Usa el entorno conda `auden-asr` (creado por setup_env.sh) y la Tesla P40.
#
# Uso:
#   ./start.sh                      # turbo_muon, sin wandb
#   ./start.sh --wandb              # turbo_muon + wandb
#   ./start.sh --optimizer adamw    # adamw puro
#   ARGS="--max-steps 50000" ./start.sh
# ============================================================================
set -euo pipefail

ENV_NAME="${AUDEN_ENV:-auden-asr}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- activar entorno ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# ---- argumentos por defecto ----
EXTRA_ARGS="${ARGS:-}"
WANDB_ARGS=""

# parsear flags propios pasados al script
while [[ $# -gt 0 ]]; do
    case "$1" in
        --wandb) WANDB_ARGS="--wandb"; shift ;;
        --optimizer) WANDB_ARGS="$WANDB_ARGS --optimizer $2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

cd "$REPO_DIR"

echo "==> Entorno: $ENV_NAME | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "==> Optimizador: ${OPTIMIZER:-turbo_muon}"
echo "==> Extras: $EXTRA_ARGS $WANDB_ARGS"

python trainer.py \
    --train-tsv datasets/cv_es/train.tsv \
    --test-tsv  datasets/cv_es/test.tsv \
    --clips-dir datasets/cv_es/clips \
    --durations datasets/cv_es/clip_durations.tsv \
    --output-dir exp/es-asr \
    --optimizer turbo_muon \
    $WANDB_ARGS \
    $EXTRA_ARGS