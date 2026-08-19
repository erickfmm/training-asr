#!/usr/bin/env bash
# ============================================================================
# start.sh — Lanza el fine-tuning ASR en español sobre Auden TTA m10.
# Usa el entorno conda `auden-asr` (creado por setup_env.sh) y la Tesla P40.
#
# Uso:
#   ./start.sh                      # turbo_muon, sin wandb
#   ./start.sh --wandb              # turbo_muon + wandb
#   ./start.sh --optimizer adamw    # adamw puro
#   ./start.sh --help               # muestra los parámetros y sus valores por defecto
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
        --help|-h)
            cat <<'EOF'
start.sh — Fine-tuning ASR en español sobre Auden TTA m10 (P40)

Uso:
  ./start.sh [--wandb] [--optimizer adamw] [ARGS de trainer.py]

Los parámetros por defecto provienen de trainer.py (defaults de argparse):

Modelo / datos:
  --model            AudenAI/auden-tta-m10
  --train-tsv        datasets/cv_es/train.tsv
  --test-tsv         datasets/cv_es/test.tsv
  --clips-dir        datasets/cv_es/clips
  --durations        datasets/cv_es/clip_durations.tsv
  --output-dir       exp/es-asr
  --resume           ""            (ruta de checkpoint; auto-resume si está vacío)
  --seed             42
  --batch-seconds    120.0         segundos de audio por batch
  --max-duration     30.0
  --num-workers      4

Entrenamiento
  --optimizer        turbo_muon    (turbo_muon | adamw)
  --lr-muon          2e-3
  --lr-adamw         3e-4
  --weight-decay     0.01
  --warmup-steps     1000
  --max-steps        20000
  --grad-clip        5.0
  --grad-accum-steps 1            (micro-batches por step del optimizador)
  --max-grad-retries 3            (fallos inf/nan antes del stop de emergencia)

Spike guard (z-score del grad norm, estilo ZClip)
  --no-spike-guard               (desactiva spike guard y snapshots de rollback)
  --spike-z          2.5          (spike moderado -> clip adaptativo)
  --spike-z-rollback 5.0          (spike severo -> rollback + salto de ventana)
  --spike-warmup     25           (steps antes de activar la detección)
  --spike-alpha      0.97         (factor EMA de las estadísticas del grad norm)
  --spike-lr-mode    temporary    (temporary | permanent | none)
  --spike-lr-factor  0.5          (reducción de LR tras rollback)
  --spike-cooldown   300          (steps con LR reducido tras rollback)
  --spike-lr-ramp    100          (steps de rampa de regreso al schedule)
  --max-rollbacks    5            (rollbacks antes del stop de emergencia)
  --snapshot-every   10           (frecuencia del snapshot 'last known good')

RNNT / decodificación
  --rnnt-warm-step   2000
  --simple-loss-scale 0.5
  --prune-range       5
  --am-scale          0.0
  --lm-scale          0.25

Checkpoints / validación
  --rolling-every    500
  --rolling-keep     3
  --val-every        500
  --val-samples      1000
  --log-every        20

wandb
  --wandb                        (activar sincronización wandb)
  --wandb-project   auden-asr-es
  --wandb-run-name  ""            (nombre del run, opcional)

Congelado
  --no-freeze-branches           (no congelar text_encoder/attention_decoder/align)

Conveniencia: ARGS="--max-steps 50000" ./start.sh pasa flags extra.
EOF
            exit 0
            ;;
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