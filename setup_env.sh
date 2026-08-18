#!/usr/bin/env bash
# ============================================================================
# setup_env.sh — Crea el entorno conda `auden-asr` compatible con Tesla P40
# (compute capability 6.1 / sm_61) para fine-tuning ASR con Auden + k2.
#
# Por qué NO se clona `frankenstein`:
#   - frankenstein usa Python 3.9; el paquete `auden` requiere Python >=3.10
#     (usa anotaciones PEP 604 `str | torch.device` sin `from __future__`).
#   - frankenstein tiene torch 2.8.0+cu128, cuyos kernels NO incluyen sm_61
#     (la P40 falla con "no kernel image available").
#
# Solución: torch 2.7.1+cu118 (incluye sm_61) + wheel k2 para torch 2.7.1/cu118.
# Driver 550 (CUDA 12.4) es retrocompatible con runtimes CUDA 11.8.
# ============================================================================
set -euo pipefail

ENV_NAME="${AUDEN_ENV:-auden-asr}"
AUDEN_REPO_DIR="${AUDEN_REPO_DIR:-$(pwd)/Auden}"

echo "==> Entorno conda: $ENV_NAME (Python 3.11)"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n "$ENV_NAME" python=3.11 -y
else
    echo "   el entorno ya existe, se reutiliza"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> torch 2.7.1 + torchaudio 2.7.1 (cu118, compatible sm_61)"
pip install --upgrade pip
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

echo "==> k2 (wheel precompilado para torch 2.7.1 + cuda 11.8 + cp311)"
pip install k2==1.24.4.dev20260625+cuda11.8.torch2.7.1 \
    --no-deps -f https://k2-fsa.github.io/k2/cuda.html

echo "==> Auden (desde GitHub, editable)"
if [ ! -d "$AUDEN_REPO_DIR" ]; then
    git clone https://github.com/AudenAI/Auden.git "$AUDEN_REPO_DIR"
fi
# auden pide numpy<2.0; forzamos 1.26 para evitar conflicto.
pip install "numpy<2.0"
pip install -e "$AUDEN_REPO_DIR"

echo "==> Dependencias extra para el trainer"
pip install soundfile jiwer pandas tqdm wandb

echo "==> Verificación rápida"
python - <<'PY'
import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| cuda disp:", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("GPU:", torch.cuda.get_device_name(0), "| sm_%d%d" % cap)
    x = torch.randn(64, 64, device="cuda")
    y = (x @ x).sum().item()
    print("matmul CUDA OK:", round(y, 3))
import k2
print("k2 OK")
import auden
print("auden import OK")
PY

echo
echo "✅ Entorno '$ENV_NAME' listo. Actívalo con:  conda activate $ENV_NAME"