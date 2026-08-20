# Experimentos de entrenamiento ASR (W&B + git)

Datos extraídos de `experimentos-wandb.csv` y cruzados con `git log` por timestamp.
Las fechas W&B están en **UTC** (`Z`); los commits se normalizaron a UTC (git local `-0400` + 4h).

---

## Resumen ejecutivo

- **9 experimentos** registrados en W&B (proyecto `auden-asr-es`), del 18 al 20 de ago 2026.
- **2 modelos** probados: `AudenAI/auden-tta-m10` (runs 1–6) y `whisper-medium` (runs 7–9).
- **1 en ejecución**, **2 failed**, **5 crashed**, **1 failed**.
- **0 con WER evaluado** — ningún run completó validación con WER (campo vacío).
- Evolución clave: `adamw`→`turbo_muon`, `lr_muon` 0.02→0.001, `warmup` 1000→5000→1000, `batch_seconds` 120→60, spike guard ZClip + rolling checkpoints + resume robusto.

---

## Tabla maestra

| Run | Modelo | Optimizer | Estado | Step final | Loss | Val loss | WER | lr_muon | lr_adamw | batch_s | warmup | max_steps | Creado (UTC) | Commit | Mensaje commit |
|-----|--------|-----------|--------|-----------|------|----------|-----|---------|----------|---------|--------|-----------|-------------|--------|----------------|
| vibrant-river-1 | auden-tta-m10 | turbo_muon | failed | 200 | 1.327 | — | — | 0.02 | 0.0003 | 120 | 1000 | 20000 | 08-18 23:04 | `e723d02` | Add Auden TTA m10 ASR fine-tuning for Spanish on P40 |
| astral-galaxy-2 | auden-tta-m10 | adamw | failed | 200 | 0.045 | — | — | 0.02 | 0.0003 | 120 | 1000 | 20000 | 08-18 23:29 | `e723d02` | (ídem) |
| smart-butterfly-3 | auden-tta-m10 | turbo_muon | crashed | 12330 | Inf | — | Inf | 0.02 | 0.0003 | 60 | 1000 | 20000 | 08-18 23:51 | `e723d02` | (ídem) |
| atomic-snowflake-4 | auden-tta-m10 | turbo_muon | crashed | — | — | — | — | 0.002 | 0.0003 | 60 | 1000 | 2e6 | 08-19 16:02 | `d18411c` | Add spike guard (ZClip rollback), grad accumulation metrics, default lr-muon 2e-3 |
| treasured-forest-5 | auden-tta-m10 | turbo_muon | failed | 110 | 0.092 | — | — | 0.002 | 0.0003 | 60 | 1000 | 2e6 | 08-19 16:03 | `d18411c` | (ídem) |
| atomic-totem-6 | auden-tta-m10 | turbo_muon | failed | 530 | 0.080 | 0.081 | 0.333 | 0.001 | 0.0002 | 60 | 5000 | 150000 | 08-19 17:08 | `d18411c` | (ídem) |
| playful-glade-7 | whisper-medium | turbo_muon | crashed | 500 | 1.569 | 1.583 | — | 0.001 | 0.0002 | 60 | 5000 | 150000 | 08-20 04:47 | `1978813` | Add --model auden\|whisper-medium with ModelBackend abstraction |
| daily-shadow-8 | whisper-medium | turbo_muon | crashed | 739 | 0.325 | 0.580 | — | 0.001 | 0.0002 | 60 | 1000 | 20000 | 08-20 16:47 | `1978813` | (ídem; crash motivó `9811f2f` Resume LR temp state) |
| asr-rolling_step-0000700 | whisper-medium | turbo_muon | **running** | 743 | 0.398 | — | — | 0.002 | 0.0001 | 60 | 1000 | 20000 | 08-20 21:34 | `b029555` | Fix wandb non-monotonic step warning on rollbacks/resume |

> **Nota:** `atomic-snowflake-4` no registró métricas (runtime 31 s — crash inmediato, probable OOM/CUDA).

---

## Aciertos principales

### 1. `asr-rolling_step-0000700` — run estable en curso (whisper-medium)
- Único experimento **running**, 743 steps, loss 0.398, 0.035 batch/s, ~22 min de runtime.
- Configuración más madura: spike guard ZClip + rolling checkpoints + resume (`rolling_step-0000700.pt`), warmup 1000, grad_accum 8, spike_cooldown 300.
- Sin NaN, sin rollbacks (0), grad_norm z=-0.62 (estable). Indica que la pila de estabilidad (commits `d18411c`+`9811f2f`+`b029555`) funciona.

### 2. `playful-glade-7` — mejor avance de whisper-medium antes de crash
- 500 steps, loss 1.57, val 1.58 (consistentes, no divergencia).
- Primer run con `whisper-medium` + `turbo_muon` + warmup 5000. Llegó lejos pero crashó (runtime 42 min).
- Confirmó que el backend `ModelBackend` (`1978813`) arranca y entrena correctamente.

### 3. `atomic-totem-6` — único run con WER medido (parcial)
- 530 steps, val_loss 0.081, **WER 0.333**. Único experimento con WER en el CSV.
- Aunque failed, demostró que auden-tta-m10 puede generar predicciones y evaluar WER.

---

## Fallas principales (categorizadas)

### A. NaN / Inf — learning rate demasiado alto
**`smart-butterfly-3`** (08-18, commit `e723d02`): `lr_muon=0.02`, sin grad accumulation → loss **Inf**, WER **Inf** a step 12330. El optimizador divergió por lr 10x sobre lo razonable.
- **Fix posterior:** commit `3d18ff9` (grad accumulation + inf/nan protection) y `d18411c` (lr_muon default 2e-3).

### B. Divergencia / colapso de loss
**`vibrant-river-1`** (08-18, `e723d02`): `lr_muon=0.02`, loss 1.33 a step 200 — no converge.
**`astral-galaxy-2`** (08-18, `e723d02`): `adamw` (no turbo_muon), loss 0.045 sospechosamente baja = **colapso** (mode collapse / trivial solution), no aprendizaje real.

### C. Crash temprano (OOM / CUDA)
**`atomic-snowflake-4`** (08-19, `d18411c`): runtime 31 s, sin métricas — falla inmediata. `max_steps=2e6` + `batch_seconds=60` + 4 workers. Probable OOM en P40.
**`treasured-forest-5`** (08-19, `d18411c`): 110 steps / 726 s, luego failed. Mismo config que snowflake-4 pero lr 0.002.

### D. Inestabilidad tardía / sobreajuste
**`atomic-totem-6`** (08-19, `d18411c`): 530 steps, val_loss 0.081 pero **WER 0.333** — brecha train/val enorme = sobreajuste. Failed tras 5h.
**`daily-shadow-8`** (08-20, `1978813`): 739 steps, val 0.58 (subiendo), crash tras 4.7h. Resume desde `rolling_step-0000500.pt` fue inestable — motivó el fix `9811f2f` (Resume LR temp state and dataloader RNG/epoch).

---

## Línea temporal: commits ↔ experimentos

```
08-18 21:01  6d2c23d  Initial commit
08-18 22:27  930cb53  Limpia info privada de investigacion.md
08-18 23:02  e723d02  Add Auden TTA m10 ASR fine-tuning (P40)
            │
            ├─ 23:04  vibrant-river-1   (failed,  200 steps, lr_muon 0.02)
            ├─ 23:29  astral-galaxy-2   (failed,  200 steps, adamw, colapso)
            └─ 23:51  smart-butterfly-3 (crashed, Inf loss, lr_muon 0.02)

08-19 15:33  3d18ff9  Add grad accumulation + inf/nan protection
08-19 15:56  d18411c  Add spike guard ZClip, default lr-muon 2e-3
08-19 15:56  0b9e5db  Add --help flag to start.sh
            │
            ├─ 16:02  atomic-snowflake-4 (crashed, 31s, OOM)
            ├─ 16:03  treasured-forest-5 (failed,  110 steps)
            └─ 17:08  atomic-totem-6     (failed,  530 steps, WER 0.333, sobreajuste)

08-20 04:43  1978813  Add --model auden|whisper-medium (ModelBackend)
            │
            ├─ 04:47  playful-glade-7  (crashed, 500 steps, loss 1.57)
            └─ 16:47  daily-shadow-8   (crashed, 739 steps, resume inestable)

08-20 20:29  9811f2f  Resume LR temp state + dataloader RNG/epoch on --resume
08-20 20:39  0a4326d  update git ignore
08-20 21:14  b029555  Fix wandb non-monotonic step warning on rollbacks/resume
            │
            └─ 21:34  asr-rolling_step-0000700 (RUNNING, 743 steps, loss 0.398)

08-20 21:55  635dbcb  Document whisper-medium backend in README
08-20 22:02  29334a2  subir exps
```

---

## Evolución de hiperparámetros

| Parámetro | Runs 1–3 (08-18) | Runs 4–6 (08-19) | Runs 7–9 (08-20) |
|-----------|-----------------|-----------------|-----------------|
| `model` | auden-tta-m10 | auden-tta-m10 | whisper-medium |
| `optimizer` | turbo_muon / adamw | turbo_muon | turbo_muon |
| `lr_muon` | 0.02 | 0.002 → 0.001 | 0.001 → 0.002 |
| `lr_adamw` | 0.0003 | 0.0003 → 0.0002 | 0.0002 → 0.0001 |
| `warmup_steps` | 1000 | 1000 → 5000 | 5000 → 1000 |
| `batch_seconds` | 120 → 60 | 60 | 60 |
| `max_steps` | 20000 | 2e6 → 150000 | 150000 → 20000 |
| `grad_accum_steps` | (no existía) | 24 | 24 → 8 |
| `spike guard` | (no existía) | ZClip | ZClip |
| `rolling ckpt` | no | no → sí | sí |
| `resume` | no | no | sí |

**Tendencia:** de configs agresivas (lr alto, sin guard) → estabilidad (lr bajo, warmup largo, grad accum, spike guard, rolling, resume).

---

## Lecciones

1. **`lr_muon=0.02` es letal** — causó Inf (smart-butterfly-3) y divergencia (vibrant-river-1). Rango seguro: 0.001–0.002.
2. **`adamw` solo no sirve** para este setup — astral-galaxy-2 colapsó a loss trivial. `turbo_muon` es el optimizador viable.
3. **Sin grad accumulation / spike guard** (runs 1–3) todo falla rápido. Los commits `3d18ff9`+`d18411c` fueron el punto de inflexión.
4. **`whisper-medium`** (runs 7–9) entrena mejor que auden-tta-m10: loss consistente, sin colapso, llega más lejos.
5. **Resume inestable** — daily-shadow-8 crashó al resumir desde `rolling_step-0000500.pt`; fix en `9811f2f` (LR temp state + RNG/epoch).
6. **Ningún run completó WER** salvo atomic-totem-6 (0.333, sobreajustado). Falta un run que termine `max_steps` y valide.
7. **OOM posible** en P40 con configs de 4 workers + batch 60s (atomic-snowflake-4, 31s). `num_workers=16` en runs 7–9 funcionó mejor.

---

## Próximos pasos sugeridos

- Dejar correr `asr-rolling_step-0000700` hasta completar 20000 steps y medir WER.
- Si WER > 0.20, considerar: más warmup (5000), unfreeze encoder de whisper, o bajar lr_adamw a 5e-5.
- Añadir evaluación WER periódica (no solo al final) para detectar sobreajuste como en atomic-totem-6.
- Documentar config óptima en `README.md` una vez que un run termine con WER < 0.15.