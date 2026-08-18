# training-asr

Fine-tuning ASR en español sobre el modelo **Auden TTA** ([`AudenAI/auden-tta-m10`](https://huggingface.co/AudenAI/auden-tta-m10), Zipformer + RNNT, 0.4B, pre-entrenado con ~4.600 h de español) usando **Common Voice es** y una **Tesla P40** (24 GB, sm_61).

Optimizador **Turbo-Muon** (con AdamW de respaldo) o AdamW puro. Logging por step en CSV, checkpoints best-3 + rolling cada 500 steps, y sincronización opcional con **Weights & Biases**.

---

## 1. Requisitos

- GPU Tesla **P40** (compute capability 6.1 / sm_61), 24 GB, driver ≥ 525 (CUDA 12.4 recomendado).
- Conda + ~20 GB libres (modelo 1.3 GB + dependencias + datasets).
- Datasets de Common Voice es en `datasets/cv_es/`:
  - `train.tsv`, `test.tsv`, `clip_durations.tsv`
  - `clips/` (mp3s)

> **Nota sobre la P40:** la P40 es sm_61. Los wheels de torch con CUDA 12.6/12.8 **no incluyen kernels sm_61**, por lo que se usa **torch 2.7.1 + cu118** (compatible). El entorno `frankenstein` original (torch 2.8.0+cu128) **no funciona** en la P40.

---

## 2. Configuración del entorno

```bash
./setup_env.sh
```

Esto crea el entorno conda `auden-asr` (Python 3.11) e instala:
- `torch==2.7.1+cu118` + `torchaudio==2.7.1+cu118` (kernels sm_61 incluidos)
- `k2` (wheel precompilado para torch 2.7.1 / cuda 11.8)
- `auden` (desde GitHub, editable)
- `soundfile`, `jiwer`, `pandas`, `tqdm`, `wandb`, `lhotse`

Verificación incluida al final del script: matmul CUDA OK en la P40, import `k2` y `auden` correctos.

---

## 3. Weights & Biases (opcional)

### 3.1 Crear cuenta y proyecto
1. Regístrate en [wandb.ai](https://wandb.ai/site).
2. Crea un proyecto nuevo llamado `auden-asr-es` (o el que prefieras).

### 3.2 Login
```bash
conda activate auden-asr
wandb login
```
Pega tu API key (la obtienes en https://wandb.ai/authorize). Se guarda en `~/.netrc`.

Alternativamente, sin login interactivo:
```bash
export WANDB_API_KEY=tu-api-key
```

### 3.3 Ejecutar con wandb
```bash
./start.sh --wandb
```
Por defecto el proyecto es `auden-asr-es`. Para cambiarlo:
```bash
python trainer.py ... --wandb --wandb-project mi-proyecto --wandb-run-name exp-01
```

> Sin `--wandb`, el entrenamiento corre con `WANDB_MODE=disabled` (sin red).

---

## 4. Ejecución

### 4.1 Inicio rápido (turbo_muon, sin wandb)
```bash
./start.sh
```

### 4.2 Con wandb
```bash
./start.sh --wandb
```

### 4.3 Con AdamW (fallback)
```bash
./start.sh --optimizer adamw
```

### 4.4 Parámetros útiles (pasar tras `--`)
```bash
./start.sh -- --max-steps 50000 --batch-seconds 200 --lr-muon 0.015
```

| Flag | Default | Descripción |
|---|---|---|
| `--optimizer` | `turbo_muon` | `turbo_muon` o `adamw` |
| `--lr-muon` | `0.02` | LR para TurboMuon (params 2D) |
| `--lr-adamw` | `3e-4` | LR para AdamW (params 1D/3D) |
| `--max-steps` | `20000` | steps totales |
| `--warmup-steps` | `1000` | warmup lineal |
| `--batch-seconds` | `120` | segundos de audio por batch (bucketing) |
| `--rolling-every` | `500` | checkpoint rolling cada N steps |
| `--rolling-keep` | `3` | conservar últimos 3 rolling (borra anteriores) |
| `--val-every` | `500` | validación + WER cada N steps |
| `--val-samples` | `1000` | clips de test para WER periódico |
| `--grad-clip` | `5.0` | clip de gradiente |
| `--resume` | auto | checkpoint a resumir (auto desde último rolling) |

---

## 5. Salidas

```
exp/es-asr/
├── train_log.csv         # métricas por step (loss, lr, grad_norm, throughput, mem)
├── val_log.csv           # métricas de validación (val_loss, wer)
├── checkpoints/
│   ├── rolling_step-0000500.pt   # rolling (últimos 3)
│   ├── rolling_step-0001000.pt
│   ├── rolling_step-0001500.pt
│   ├── best_step-0000500_wer-0.1234.pt  # best-3 por WER
│   └── best_meta.json
├── best_model/           # export del mejor modelo (inferencia)
└── final_model/          # export del modelo final
```

### CSV `train_log.csv` (por step)
`step, epoch, loss, simple_loss, pruned_loss, s_scale, p_scale, lr_muon, lr_adamw, grad_norm, clip_ratio, frames, batch_size, tok_per_sec, batches_per_sec, mem_alloc_gb, mem_reserved_gb, elapsed_sec`

### CSV `val_log.csv` (cada `val_every`)
`step, val_loss, wer`

---

## 6. Reanudar

El entrenamiento **auto-resume** desde el checkpoint rolling más reciente si existe. Para uno específico:
```bash
python trainer.py ... --resume exp/es-asr/checkpoints/rolling_step-0001000.pt
```

---

## 7. Optimizadores

### Turbo-Muon (default)
[Turbo-Muon](https://arxiv.org/abs/2512.04632) (Boissin et al. 2025) precondiciona el gradiente con AOL antes de las iteraciones de Newton-Schulz. En la P40 (sin GEMM bf16) las iteraciones NS se ejecutan en **float32**. Se aplica a parámetros 2D (matrices ocultas); AdamW maneja el resto (biases, normas, convoluciones 3D, embeddings).

### AdamW (fallback)
```bash
./start.sh --optimizer adamw
```

---

## 8. Arquitectura del entrenamiento

- **Modelo:** `AudenAI/auden-tta-m10` (Zipformer encoder + RNNT decoder/joiner).
- **Ramas congeladas:** text_encoder (BERT), attention_decoder, heads de align (no necesarias para ASR transcribe). Sólo entrena encoder + decoder RNNT + joiner + proyecciones (~137M params).
- **Features:** fbank 80 bins on-the-fly (dither=0, snip_edges=False), igual que Auden.
- **Pérdida:** RNNT pruned (simple + pruned) con warmup de escalas (receta oficial Auden).
- **Datos:** Common Voice es, decodificación mp3 con soundfile/libsndfile, resample 16 kHz, bucketing por duración.
- **dtype:** fp32 (la P40 no tiene tensor cores; AMP fp16 es lento/inestable).

---

## 9. Monitoreo

```bash
# ver progreso en vivo del CSV
tail -f exp/es-asr/train_log.csv

# en wandb
# https://wandb.ai/<tu-usuario>/auden-asr-es
```

---

## 10. Resolución de problemas

| Problema | Solución |
|---|---|
| `no kernel image available for execution on the device` | torch no soporta sm_61. Reinstala con `torch==2.7.1+cu118` (ver `setup_env.sh`). |
| `k2` no encuentra wheel compatible | verifica `python -c "import torch; print(torch.__version__)"` sea `2.7.1+cu118`. |
| mp3 no carga | `soundfile` requiere libsndfile ≥ 1.1. `pip install --upgrade soundfile`. |
| OOM (24 GB) | reduce `--batch-seconds` (ej. 80) o `--max-duration` (ej. 20). |
| k2 RNNT loss lento | normal en P40; sube `--batch-seconds` para mejor throughput. |