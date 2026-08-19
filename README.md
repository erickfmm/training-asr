# training-asr

Fine-tuning ASR en español sobre el modelo **Auden TTA** ([`AudenAI/auden-tta-m10`](https://huggingface.co/AudenAI/auden-tta-m10), Zipformer + RNNT, 0.4B, pre-entrenado con ~4.600 h de español) usando **Common Voice es** y una **Tesla P40** (24 GB, sm_61).

Optimizador **Turbo-Muon** (con AdamW de respaldo) o AdamW puro. Logging por step en CSV, checkpoints best-3 + rolling cada 500 steps, y sincronización opcional con **Weights & Biases**.

Incluye **grad accumulation** y un sistema de **estabilidad de entrenamiento** en tres capas (inf/nan, spike guard estilo ZClip con rollback, y clip adaptativo) para evitar divergencias — ver [§ 9](#9-estabilidad-del-entrenamiento).

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
| `--lr-muon` | `2e-3` | LR para TurboMuon (params 2D) |
| `--lr-adamw` | `3e-4` | LR para AdamW (params 1D/3D) |
| `--max-steps` | `20000` | steps totales |
| `--warmup-steps` | `1000` | warmup lineal |
| `--batch-seconds` | `120` | segundos de audio por batch (bucketing) |
| `--rolling-every` | `500` | checkpoint rolling cada N steps |
| `--rolling-keep` | `3` | conservar últimos 3 rolling (borra anteriores) |
| `--val-every` | `500` | validación + WER cada N steps |
| `--val-samples` | `1000` | clips de test para WER periódico |
| `--grad-clip` | `5.0` | clip de gradiente fijo |
| `--resume` | auto | checkpoint a resumir (auto desde último rolling) |

### Parámetros de estabilidad

#### Grad accumulation & protección inf/nan

| Flag | Default | Descripción |
|---|---|---|
| `--grad-accum-steps` | `1` | micro-batches por step del optimizador (grad accumulation) |
| `--max-grad-retries` | `3` | fallos inf/nan **consecutivos** antes del stop de emergencia |

#### Spike guard (estilo ZClip) + rollback

| Flag | Default | Descripción |
|---|---|---|
| `--spike-z` | `2.5` | z-score para spike moderado (clip adaptativo, tier 1) |
| `--spike-z-rollback` | `5.0` | z-score para spike severo (rollback + salto de ventana, tier 2) |
| `--spike-warmup` | `25` | steps antes de activar la detección |
| `--spike-alpha` | `0.97` | factor EMA de las estadísticas del grad norm |
| `--spike-lr-mode` | `temporary` | reducción de LR tras rollback: `temporary`, `permanent` o `none` |
| `--spike-lr-factor` | `0.5` | factor de reducción temporal de LR |
| `--spike-cooldown` | `300` | steps con LR reducido (modo `temporary`) |
| `--spike-lr-ramp` | `100` | steps de rampa lineal de regreso al schedule |
| `--max-rollbacks` | `5` | rollbacks máximos antes del stop de emergencia |
| `--snapshot-every` | `10` | frecuencia (steps) del snapshot "last known good" en RAM |
| `--no-spike-guard` | — | desactiva el spike guard y los snapshots de rollback |

> El `--lr-muon` por defecto es **`2e-3`**: `0.02` es el LR típico de Muon para **pre-training desde cero** (nanoGPT speedrun); para **fine-tuning** de un modelo ya pre-entrenado es muy alto y dispara spikes (ver [§ 9](#9-estabilidad-del-entrenamiento)). Si entrenas desde cero puedes subirlo, pero bárrelo antes.

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
`step, epoch, loss, simple_loss, pruned_loss, s_scale, p_scale, lr_muon, lr_adamw, grad_norm, clip_ratio, frames, batch_size, tok_per_sec, batches_per_sec, mem_alloc_gb, mem_reserved_gb, elapsed_sec, nan_failures, lr_scale, gn_z, gn_mu, gn_sigma, spike_tier, zclip_cap, rollbacks, lr_temp_scale`

- `nan_failures`: fallos inf/nan consecutivos (step exitoso lo resetea).
- `lr_scale`: `0.5 ** lr_halvings` (factor acumulado de halvings permanentes por inf/nan).
- `gn_z`, `gn_mu`, `gn_sigma`: z-score y EMAs de la media/desviación del grad norm pre-clip.
- `spike_tier`: `0` normal, `1` spike moderado (clip adaptativo), `2` spike severo (rollback).
- `zclip_cap`: norma objetivo aplicada por el clip adaptativo del tier 1 (`-1` si no se aplicó).
- `rollbacks`: contador acumulado de rollbacks por spike severo.
- `lr_temp_scale`: multiplicador de LR temporal vigente (1.0 fuera de cooldown).

### CSV `val_log.csv` (cada `val_every`)
`step, val_loss, wer`

---

## 6. Estabilidad del entrenamiento

El entrenamiento puede divergir por spikes de gradiente (normas hasta ~1000× lo típico) y valores inf/nan. Se implementan **tres capas** de protección, en este orden en cada step:

### 1. Grad accumulation + guard inf/nan (`--grad-accum-steps`, `--max-grad-retries`)
- Los batches se agrupan en ventanas de `--grad-accum-steps` micro-batches; cada micro hace `backward()` escalado por `1/nf/len(micros)` y hay **un solo** `opt.step()` por ventana.
- Antes del step se verifica que la loss y los gradientes sean finitos (`grads_have_inf_nan`).
- Si hay inf/nan: se descartan los gradientes, se **reduce el LR a la mitad de forma permanente** (`scheduler.scale_base(0.5)`) y se **reintenta la misma ventana**. Tras `--max-grad-retries` fallos **consecutivos** (un step exitoso resetea el contador) se guarda un checkpoint de emergencia y se detiene el run (`exit 1`).

2. **Spike guard estilo ZClip** (`--spike-z`, `--spike-z-rollback`, etc.)
- Se mantienen EMAs (`--spike-alpha`) de la **media μ y desviación σ del grad norm pre-clip**, inicializadas con una ventana de `--spike-warmup` steps.
- **Tier 1** (`z > --spike-z`): spike moderado → **clip adaptativo** reciproqual a `μ + (z²/z)·σ` (además del clip fijo `--grad-clip`). Entrenamiento continúa sin interrumpir.
- **Tier 2** (`z > --spike-z-rollback`): spike severo → **rollback**: se restaura el último snapshot sano (`--snapshot-every` steps, guardado en RAM CPU), se **salta la ventana ofensora** (receta PaLM/GLM-130B: el spike no suele reproducirse al reprocesar el mismo batch), se reduce el LR **temporalmente** (`--spike-lr-factor` durante `--spike-cooldown` steps + rampa `--spike-lr-ramp`; o `permanent`/`none`), y el contador de step retrocede.
- Tras `--max-rollbacks` rollbacks → checkpoint de emergencia y detención (`reason: spike_rollbacks`).

3. **Clip de gradiente fijo** (`--grad-clip`): backstop para el resto.

> Los valores spiky **no contaminan** las estadísticas del tracker (el tier 2 se excluye y el tier 1 se aporta capado). El estado del tracker, `lr_halvings` y `rollbacks` se guardan en todos los checkpoints, por lo que un resume no pierde la protección ni los halvings permanentes.

### Por qué los spikes y el LR por defecto

- **Muon/TurboMuon es inestable a escala**: Kimi K2 (arXiv:2507.20534) documentó loss spikes con Muon *vanilla* y tuvo que inventar **MuonClip** (cap de attention logits) para entrenar 15.5T tokens con *cero* spikes. En experimentos mid-scale MuonClip eliminó spikes que Muon sí mostraba.
- **El clipping fijo falla**: la distribución de grad norms deriva durante el entrenamiento, así que un umbral constante sub/sobre-recorta (ZClip, arXiv:2504.02507; spikes hasta 1000× lo típico).
- **`--lr-muon` en fine-tuning**: `0.02` es el LR de Muon para *pre-training desde cero*; para fine-tuning de un modelo pre-entrenado el LR óptimo baja y `0.02` dispara spikes. El default es ahora **`2e-3`** (el blog de PyTorch/DeepSpeed fine-tuneando Moonlight-16B usó `1e-4`). Si ves spikes, baja el LR y/o sube `--spike-warmup` si aparecen temprano.

---

## 7. Reanudar

El entrenamiento **auto-resume** desde el checkpoint rolling más reciente si existe. Para uno específico:
```bash
python trainer.py ... --resume exp/es-asr/checkpoints/rolling_step-0001000.pt
```

---

## 8. Optimizadores

### Turbo-Muon (default)
[Turbo-Muon](https://arxiv.org/abs/2512.04632) (Boissin et al. 2025) precondiciona el gradiente con AOL antes de las iteraciones de Newton-Schulz. En la P40 (sin GEMM bf16) las iteraciones NS se ejecutan en **float32**. Se aplica a parámetros 2D (matrices ocultas); AdamW maneja el resto (biases, normas, convoluciones 3D, embeddings).

### AdamW (fallback)
```bash
./start.sh --optimizer adamw
```

---

## 9. Arquitectura del entrenamiento

- **Modelo:** `AudenAI/auden-tta-m10` (Zipformer encoder + RNNT decoder/joiner).
- **Ramas congeladas:** text_encoder (BERT), attention_decoder, heads de align (no necesarias para ASR transcribe). Sólo entrena encoder + decoder RNNT + joiner + proyecciones (~137M params).
- **Features:** fbank 80 bins on-the-fly (dither=0, snip_edges=False), igual que Auden.
- **Pérdida:** RNNT pruned (simple + pruned) con warmup de escalas (receta oficial Auden).
- **Datos:** Common Voice es, decodificación mp3 con soundfile/libsndfile, resample 16 kHz, bucketing por duración.
- **dtype:** fp32 (la P40 no tiene tensor cores; AMP fp16 es lento/inestable).

---

## 10. Monitoreo

```bash
# ver progreso en vivo del CSV
tail -f exp/es-asr/train_log.csv

# en wandb
# https://wandb.ai/<tu-usuario>/auden-asr-es
```

---

## 11. Resolución de problemas

| Problema | Solución |
|---|---|
| `no kernel image available for execution on the device` | torch no soporta sm_61. Reinstala con `torch==2.7.1+cu118` (ver `setup_env.sh`). |
| `k2` no encuentra wheel compatible | verifica `python -c "import torch; print(torch.__version__)"` sea `2.7.1+cu118`. |
| mp3 no carga | `soundfile` requiere libsndfile ≥ 1.1. `pip install --upgrade soundfile`. |
| OOM (24 GB) | reduce `--batch-seconds` (ej. 80) o `--max-duration` (ej. 20). |
| k2 RNNT loss lento | normal en P40; sube `--batch-seconds` para mejor throughput. |