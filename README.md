# training-asr

Fine-tuning **ASR en español** con **Common Voice es** sobre una **Tesla P40** (24 GB, sm_61), con dos backends intercambiables vía la abstracción `ModelBackend`:

- **`auden`** — [`AudenAI/auden-tta-m10`](https://huggingface.co/AudenAI/auden-tta-m10) (Zipformer + RNNT, ~414 M params, pre-entrenado con ~4.600 h de español).
- **`whisper-medium`** — [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium) (encoder-decoder seq2seq + CE, ~764 M params; se baja la última revisión de HF Hub).

El **bucle de entrenamiento, optimizadores híbridos (Turbo-Muon + AdamW), warmup-coseno, grad-accum, spike-guard/rollback y checkpointing son idénticos para ambos backends**; sólo difieren la carga del modelo, las features, la pérdida y el decode (`trainer.py:262`).

Optimizador **Turbo-Muon** (con AdamW de respaldo) o **AdamW** puro. Logging por step en CSV, checkpoints best-3 + rolling cada 500 steps, y sincronización opcional con **Weights & Biases**.

Incluye **grad accumulation** y un sistema de **estabilidad de entrenamiento** en tres capas (inf/nan, spike guard estilo ZClip con rollback, y clip adaptativo) para evitar divergencias — ver [§ 9](#9-estabilidad-del-entrenamiento).

---

## 1. Requisitos

- GPU Tesla **P40** (compute capability 6.1 / sm_61), 24 GB, driver ≥ 525 (CUDA 12.4 recomendado).
- Conda + ~20 GB libres (modelo + dependencias + datasets). Whisper-medium pesa ~3 GB en HF Hub.
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
- `transformers` (para el backend `whisper-medium`)
- `soundfile`, `jiwer`, `pandas`, `tqdm`, `wandb`, `lhotse`

Verificación incluida al final del script: matmul CUDA OK en la P40, import `k2` y `auden` correctos.

---

## 3. Modelos

| | `auden` (default) | `whisper-medium` |
|---|---|---|
| Modelo HF | `AudenAI/auden-tta-m10` | `openai/whisper-medium` |
| Arquitectura | Zipformer encoder + RNNT decoder/joiner | Encoder-decoder seq2seq |
| Params totales | ~414 M | ~764 M |
| Features | fbank 80 bins on-the-fly (dither=0, snip_edges=False) | `WhisperFeatureExtractor` (log-mel 80, ventana 30 s, hop 160, 16 kHz) |
| Pérdida | RNNT pruned (`simple_loss` + `pruned_loss`, warmup de escalas) | Cross-entropy sobre token logits |
| Congelado por defecto | `text_encoder` (BERT), `attention_decoder`, heads de `align`/`s2t` | `model.encoder.*` (decoder + `proj_out` entrenables) |
| Params entrenables (default) | ~199 M (48 %) | ~457 M (60 %; decoder + proj_out) |
| Decode | `model.generate((x, lens), task="transcribe")` | `model.generate(language="es", task="transcribe")` |
| Flags de pérdida | `--rnnt-warm-step`, `--simple-loss-scale`, `--prune-range`, `--am-scale`, `--lm-scale` | (ignoradas; el trainer avisa por warning) |
| `--output-dir` por defecto | `exp/es-asr` | `exp/es-whisper-medium` |
| Normalización de texto | minúsculas + quita `¿¡` (`normalize_text`) | sin normalizar (whisper aprende case+puntuación; WER contra refs crudos) |

### Congelado

- **Auden:** sólo entrena el encoder + decoder RNNT + joiner + proyecciones. Usa `--no-freeze-branches` para no congelar nada.
- **Whisper-medium:** encoder congelado por defecto siguiendo **Vividh-ASR** (arXiv:2605.13087 — adaptar el decoder preserva la geometría acústica del encoder e iguala/supera el full fine-tune en Common Voice) y **Gumbel-BEARD** (arXiv:2606.11429 — SOTA con Whisper-medium adaptando poco el encoder; 10 h etiquetados igualan un baseline supervisado de 133 h). Usa `--whisper-unfreeze-encoder` para full fine-tune.

> **Continual learning** (arXiv:2407.03645): freeze + LR re-scaling al adaptar Whisper a lenguas no vistas de Common Voice.

---

## 4. Weights & Biases (opcional)

### 4.1 Crear cuenta y proyecto
1. Regístrate en [wandb.ai](https://wandb.ai/site).
2. Crea un proyecto nuevo llamado `auden-asr-es` (o el que prefieras).

### 4.2 Login
```bash
conda activate auden-asr
wandb login
```
Pega tu API key (la obtienes en https://wandb.ai/authorize). Se guarda en `~/.netrc`.

Alternativamente, sin login interactivo:
```bash
export WANDB_API_KEY=tu-api-key
```

### 4.3 Ejecutar con wandb
```bash
./start.sh --wandb
```
Por defecto el proyecto es `auden-asr-es`. Para cambiarlo:
```bash
python trainer.py ... --wandb --wandb-project mi-proyecto --wandb-run-name exp-01
```

> Sin `--wandb`, el entrenamiento corre con `WANDB_MODE=disabled` (sin red).

> El step que ve wandb (`wandb_step`) es **monotónico y separado** del step del optimizador, de modo que no retrocede al hacer un rollback, reintentar inf/nan o resumir (esto evita el warning de wandb "non-monotonic step").

---

## 5. Ejecución

### 5.1 Inicio rápido
```bash
./start.sh                                  # auden + turbo_muon, sin wandb
./start.sh --model whisper-medium           # whisper-medium (encoder congelado)
./start.sh --model whisper-medium --wandb
./start.sh --model auden --optimizer adamw  # auden con AdamW
```

### 5.2 Flags propios de `start.sh`
| Flag | Descripción |
|---|---|
| `--model` | `auden` (default) \| `whisper-medium` |
| `--optimizer` | `turbo_muon` (default) \| `adamw` |
| `--wandb` | activa sincronización wandb |
| `--help` / `-h` | imprime **todos** los parámetros de `trainer.py` con sus defaults |

### 5.3 Pasar flags extra a `trainer.py`
`start.sh` no trata `--` de forma especial; los flags extra se inyectan vía la variable de entorno `ARGS` (que el script reenvía tal cual):
```bash
ARGS="--max-steps 50000 --batch-seconds 200 --lr-muon 0.015" ./start.sh
ARGS="--model whisper-medium --whisper-unfreeze-encoder" ./start.sh
```

### 5.4 Parámetros principales

> La lista canónica completa está en `./start.sh --help` (se mantiene en sincronía con los defaults de argparse de `trainer.py`).

| Flag | Default | Descripción |
|---|---|---|
| `--model` | `auden` | `auden` \| `whisper-medium` |
| `--optimizer` | `turbo_muon` | `turbo_muon` o `adamw` (aplicable a ambos modelos) |
| `--lr-muon` | `2e-3` | LR para TurboMuon (params 2D) |
| `--lr-adamw` | `3e-4` | LR para AdamW (params 1D/3D) |
| `--max-steps` | `20000` | steps totales |
| `--warmup-steps` | `1000` | warmup lineal |
| `--batch-seconds` | `120` | segundos de audio por batch (bucketing) |
| `--max-duration` | `30.0` | descarta clips > N segundos |
| `--num-workers` | `4` | workers del dataset (no usado por el bucketing síncrono, pero reservado) |
| `--rolling-every` | `500` | checkpoint rolling cada N steps |
| `--rolling-keep` | `3` | conservar últimos 3 rolling (borra anteriores) |
| `--val-every` | `500` | validación + WER cada N steps |
| `--val-samples` | `1000` | clips de test para WER periódico |
| `--grad-clip` | `5.0` | clip de gradiente fijo |
| `--resume` | auto | checkpoint a resumir (auto desde último rolling) |
| `--seed` | `42` | semilla (random, numpy, torch) |

### 5.5 Whisper-medium (sólo `--model whisper-medium`)

| Flag | Default | Descripción |
|---|---|---|
| `--whisper-unfreeze-encoder` | — | no congelar el encoder (full fine-tune; por defecto congelado) |
| `--whisper-language` | `es` | idioma pasado a `model.generate` |
| `--whisper-task` | `transcribe` | tarea para `model.generate` |

> Los flags RNNT/auden-only (`--rnnt-warm-step`, `--simple-loss-scale`, `--prune-range`, `--am-scale`, `--lm-scale`, `--no-freeze-branches`) **son ignorados** con `--model whisper-medium`; el trainer emite un warning listándolos.

### 5.6 Parámetros de estabilidad

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

### 5.7 wandb

| Flag | Default | Descripción |
|---|---|---|
| `--wandb` | — | activar sincronización wandb |
| `--wandb-project` | `auden-asr-es` | proyecto wandb |
| `--wandb-run-name` | `""` | nombre del run (opcional) |
| `--wandb-resume-id` | `""` | ID de run wandb para resumir (`resume='allow'`); si vacío, se deriva de `--resume` (ver [§ 7](#7-reanudar)) |

---

## 6. Salidas

```
exp/<output-dir>/            # exp/es-asr (auden) | exp/es-whisper-medium (whisper-medium)
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

- El **export** del modelo (`backend.export`) difiere por backend: auden guarda `model.pt` (+ `config` si está disponible); whisper-medium usa `save_pretrained` del modelo + processor (formato HF estándar, listo para `WhisperForConditionalGeneration.from_pretrained`).
- El `CSVLogger` detecta si el CSV existente tiene un **header viejo** (columnas distintas) y lo renombra a `train_log.old-<timestamp>.csv` antes de empezar uno nuevo, en lugar de corruptar filas.

### CSV `train_log.csv` (por step)
`step, epoch, loss, simple_loss, pruned_loss, s_scale, p_scale, lr_muon, lr_adamw, grad_norm, clip_ratio, frames, batch_size, tok_per_sec, batches_per_sec, mem_alloc_gb, mem_reserved_gb, elapsed_sec, nan_failures, lr_scale, gn_z, gn_mu, gn_sigma, spike_tier, zclip_cap, rollbacks, lr_temp_scale`

- `nan_failures`: fallos inf/nan consecutivos (step exitoso lo resetea).
- `lr_scale`: `0.5 ** lr_halvings` (factor acumulado de halvings permanentes por inf/nan).
- `gn_z`, `gn_mu`, `gn_sigma`: z-score y EMAs de la media/desviación del grad norm pre-clip.
- `spike_tier`: `0` normal, `1` spike moderado (clip adaptativo), `2` spike severo (rollback).
- `zclip_cap`: norma objetivo aplicada por el clip adaptativo del tier 1 (`-1` si no se aplicó).
- `rollbacks`: contador acumulado de rollbacks por spike severo.
- `lr_temp_scale`: multiplicador de LR temporal vigente (1.0 fuera de cooldown).
- Con `whisper-medium`, `simple_loss`/`pruned_loss`/`s_scale`/`p_scale` se rellenan como réplicas/constantes de la CE (la pérdida efectiva es `loss`); los `frames` se fijan a 1 porque la CE de whisper ya está normalizada por tokens.

### CSV `val_log.csv` (cada `val_every`)
`step, val_loss, wer`

---

## 7. Reanudar

El entrenamiento **auto-resume** desde el checkpoint rolling más reciente si existe. Para uno específico:
```bash
python trainer.py ... --resume exp/es-asr/checkpoints/rolling_step-0001000.pt
```

### Estado preservado en checkpoints

Además de pesos del modelo y estado de los optimizadores, los checkpoints guardan y restauran:

| Campo | Efecto al reanudar |
|---|---|
| `grad_tracker` (`mu`, `sigma`, `n`) | El spike guard sigue con las mismas EMAs (no reinicia el warmup). |
| `lr_halvings` | Se **re-aplican** los halvings permanentes de LR acumulados por inf/nan (`scheduler.scale_base(0.5 ** lr_halvings)`). |
| `rollbacks` | Contador acumulado de rollbacks (para el límite `--max-rollbacks`). |
| `scheduler_temp` | El **LR temporal** (cooldown + rampa tras un rollback) se restaura, de modo que un checkpoint tomado durante la reducción no la pierde al reanudar. |
| `rng_state` + `epoch` | El RNG del dataloader se restaura en el punto exacto del checkpoint; los batches de la época interrumpida que no se consumieron **se saltan** (no se repite desde el inicio de la época). |
| `best_wer` | Se conserva el mejor WER histórico para el best-3. |

### Checkpoints legacy

Si el checkpoint no tiene `rng_state` (formato anterior), el trainer **aproxima la época** leyendo el último `epoch` de `train_log.csv` y deriva un RNG determinista `seed + epoch` que no repite desde el principio. Se loguea como `Checkpoint legacy: época aproximada desde CSV = N`.

### wandb resume

- `--wandb-resume-id <id>` tiene prioridad y fuerza `resume='allow'` sobre esa run.
- Si se omite y hay `--resume`, el ID se **deriva deterministamente** del nombre del checkpoint (`"asr-" + <nombre-del-ckpt>`), de modo que reanudar el mismo archivo continúa la misma run en lugar de arrancar desde step 0.
- El `wandb_step` es **monotónico**: al reanudar arranca por delante del último step logueado en wandb (o, como red de seguridad, por delante del step del optimizador reanudado), evitando el warning "non-monotonic step".

---

## 8. Optimizadores

### Turbo-Muon (default)
[Turbo-Muon](https://arxiv.org/abs/2512.04632) (Boissin et al. 2025) precondiciona el gradiente con AOL antes de las iteraciones de Newton-Schulz. En la P40 (sin GEMM bf16) las iteraciones NS se ejecutan en **float32**. Se aplica a parámetros 2D (matrices ocultas); AdamW maneja el resto (biases, normas, convoluciones 3D, embeddings). Funciona con ambos backends.

### AdamW (fallback)
```bash
./start.sh --optimizer adamw
```

---

## 9. Estabilidad del entrenamiento

El entrenamiento puede divergir por spikes de gradiente (normas hasta ~1000× lo típico) y valores inf/nan. Se implementan **tres capas** de protección, en este orden en cada step:

### 1. Grad accumulation + guard inf/nan (`--grad-accum-steps`, `--max-grad-retries`)
- Los batches se agrupan en ventanas de `--grad-accum-steps` micro-batches; cada micro hace `backward()` escalado por `1/nf/len(micros)` y hay **un solo** `opt.step()` por ventana.
- Antes del step se verifica que la loss y los gradientes sean finitos (`grads_have_inf_nan`).
- Si hay inf/nan: se descartan los gradientes, se **reduce el LR a la mitad de forma permanente** (`scheduler.scale_base(0.5)`) y se **reintenta la misma ventana**. Tras `--max-grad-retries` fallos **consecutivos** (un step exitoso resetea el contador) se guarda un checkpoint de emergencia y se detiene el run (`exit 1`).

### 2. Spike guard estilo ZClip (`--spike-z`, `--spike-z-rollback`, etc.)
- Se mantienen EMAs (`--spike-alpha`) de la **media μ y desviación σ del grad norm pre-clip**, inicializadas con una ventana de `--spike-warmup` steps.
- **Tier 1** (`z > --spike-z`): spike moderado → **clip adaptativo** reciproqual a `μ + (z²/z)·σ` (además del clip fijo `--grad-clip`). Entrenamiento continúa sin interrumpir.
- **Tier 2** (`z > --spike-z-rollback`): spike severo → **rollback**: se restaura el último snapshot sano (`--snapshot-every` steps, guardado en RAM CPU), se **salta la ventana ofensora** (receta PaLM/GLM-130B: el spike no suele reproducirse al reprocesar el mismo batch), se reduce el LR **temporalmente** (`--spike-lr-factor` durante `--spike-cooldown` steps + rampa `--spike-lr-ramp`; o `permanent`/`none`), y el contador de step retrocede.
- Tras `--max-rollbacks` rollbacks → checkpoint de emergencia y detención (`reason: spike_rollbacks`).

### 3. Clip de gradiente fijo (`--grad-clip`): backstop para el resto.

> Los valores spiky **no contaminan** las estadísticas del tracker (el tier 2 se excluye y el tier 1 se aporta capado). El estado del tracker, `lr_halvings`, `rollbacks`, el `scheduler_temp` y el `rng_state` se guardan en todos los checkpoints, por lo que un resume no pierde la protección ni los halvings permanentes.

### Por qué los spikes y el LR por defecto

- **Muon/TurboMuon es inestable a escala**: Kimi K2 (arXiv:2507.20534) documentó loss spikes con Muon *vanilla* y tuvo que inventar **MuonClip** (cap de attention logits) para entrenar 15.5T tokens con *cero* spikes. En experimentos mid-scale MuonClip eliminó spikes que Muon sí mostraba.
- **El clipping fijo falla**: la distribución de grad norms deriva durante el entrenamiento, así que un umbral constante sub/sobre-recorta (ZClip, arXiv:2504.02507; spikes hasta 1000× lo típico).
- **`--lr-muon` en fine-tuning**: `0.02` es el LR de Muon para *pre-training desde cero*; para fine-tuning de un modelo pre-entrenado el LR óptimo baja y `0.02` dispara spikes. El default es ahora **`2e-3`** (el blog de PyTorch/DeepSpeed fine-tuneando Moonlight-16B usó `1e-4`). Si ves spikes, baja el LR y/o sube `--spike-warmup` si aparecen temprano.

---

## 10. Arquitectura del entrenamiento

### Común a ambos backends
- **dtype NS:** fp32 en P40 (sm<80); bfloat16 en GPUs sm≥80. Aplica a las iteraciones de Newton-Schulz de TurboMuon.
- **Datos:** Common Voice es, decodificación mp3 con soundfile/libsndfile, resample 16 kHz, `CVesDataset` reutilizado por ambos backends (misma sr).
- **Bucketing:** por duración con padding dinámico (bloques de ~`batch_seconds/2`, batches ≤32 clips), batches homogéneos para menos padding.
- **LR schedule:** warmup lineal + coseno (`WarmupCosine`), con soporte para reducción temporal (cooldown + rampa) tras un rollback.
- **Bucle, optimizadores, spike-guard, checkpointing, logging:** idénticos (ver [§ 9](#9-estabilidad-del-entrenamiento)).

### `auden`
- **Modelo:** `AudenAI/auden-tta-m10` (Zipformer encoder + RNNT decoder/joiner).
- **Ramas congeladas:** `text_encoder` (BERT), `attention_decoder`, heads de `align`/`s2t` (no necesarias para ASR transcribe). Sólo entrena encoder + decoder RNNT + joiner + proyecciones (~199 M params).
- **Features:** fbank 80 bins on-the-fly (dither=0, snip_edges=False), igual que Auden.
- **Pérdida:** RNNT pruned (`simple_loss` + `pruned_loss`) con warmup de escalas (receta oficial Auden).

### `whisper-medium`
- **Modelo:** `openai/whisper-medium` (encoder-decoder seq2seq, CE sobre token logits).
- **Congelado:** encoder por defecto (Vividh-ASR/Gumbel-BEARD); decoder + `proj_out` entrenables (~457 M params). `--whisper-unfreeze-encoder` para full fine-tune (~764 M).
- **Features:** `WhisperFeatureExtractor` (log-mel 80, ventana 30 s, hop 160, 16 kHz).
- **Pérdida:** cross-entropy media sobre tokens no-padding (los `pad_token_id` se enmascaran con `-100`); `nf=1` porque la CE ya está normalizada por tokens y el grad-accum la reparte correctamente.

### Literatura
- **Vividh-ASR** (arXiv:2605.13087): adaptar el decoder preserva la geometría acústica del encoder e iguala/supera el full fine-tune en Common Voice.
- **Gumbel-BEARD** (arXiv:2606.11429): SOTA con Whisper-medium adaptando poco el encoder; 10 h etiquetados igualan un baseline supervisado de 133 h.
- **Continual-learning** (arXiv:2407.03645): freeze + LR re-scaling al adaptar Whisper a lenguas no vistas de Common Voice.

---

## 11. Monitoreo

```bash
# ver progreso en vivo del CSV
tail -f exp/es-asr/train_log.csv          # o exp/es-whisper-medium/train_log.csv

# en wandb
# https://wandb.ai/<tu-usuario>/auden-asr-es
```

---

## 12. Resolución de problemas

| Problema | Solución |
|---|---|
| `no kernel image available for execution on the device` | torch no soporta sm_61. Reinstala con `torch==2.7.1+cu118` (ver `setup_env.sh`). |
| `k2` no encuentra wheel compatible | verifica `python -c "import torch; print(torch.__version__)"` sea `2.7.1+cu118`. |
| mp3 no carga | `soundfile` requiere libsndfile ≥ 1.1. `pip install --upgrade soundfile`. |
| OOM (24 GB) con `auden` | reduce `--batch-seconds` (ej. 80) o `--max-duration` (ej. 20). |
| OOM con `whisper-medium` | el encoder congelado consume menos; si usas `--whisper-unfreeze-encoder` y te quedas sin memoria, baja `--batch-seconds` a 60–80. |
| `k2` RNNT loss lento (auden) | normal en P40; sube `--batch-seconds` para mejor throughput. |
| `from transformers import ...` falla (whisper) | `pip install transformers` dentro del entorno `auden-asr`. |
| warning wandb `non-monotonic step` | ya resuelto: el `wandb_step` es monotónico y separado del step del optimizador (commit `b029555`). Si reaparece tras un resume, usa `--wandb-resume-id`. |
| `--model whisper-medium` ignora `--rnnt-warm-step`, etc. | esperado: los flags RNNT/auden-only se ignoran y se loguean como warning. |
| spikes de grad norm frecuentes | baja `--lr-muon` (ej. 1e-3) y/o sube `--spike-warmup` (ej. 50) si aparecen temprano; ver [§ 9](#9-estabilidad-del-entrenamiento). |