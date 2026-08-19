#!/usr/bin/env python
"""Fine-tuning ASR en español sobre Auden TTA (AudenAI/auden-tta-m10) con
Common Voice es, optimizado para Tesla P40 (sm_61, 24 GB, fp32).

Características:
  - Optimizador Turbo-Muon (híbrido con AdamW) o AdamW puro (--optimizer).
  - Grad accumulation (--grad-accum-steps) + protección inf/nan: descarta la
    ventana afectada, reduce el LR a la mitad (permanente) y reintenta; tras
    3 fallos consecutivos guarda checkpoint de emergencia y detiene el run.
  - Spike guard estilo ZClip: z-score del grad norm con EMAs; spikes moderados
    (z>2.5) se recortan con clip adaptativo; spikes severos (z>5) hacen
    rollback al último snapshot sano (cada N steps), reducen el LR
    temporalmente y saltan la ventana afectada (receta PaLM/GLM-130B).
  - Logging por step en CSV (loss, lr real, norma del gradiente, throughput, mem).
  - Checkpoints rolling cada 500 steps (conserva los últimos 3) + best-3 por WER.
  - Sincronización opcional con Weights & Biases (--wandb).
  - Sólo entrena la rama RNNT ASR (congela text_encoder/attention_decoder/align).

Uso típico:
    python trainer.py \
        --train-tsv datasets/cv_es/train.tsv \
        --test-tsv  datasets/cv_es/test.tsv \
        --clips-dir datasets/cv_es/clips \
        --durations datasets/cv_es/clip_durations.tsv \
        --output-dir exp/es-asr \
        --optimizer turbo_muon \
        --max-steps 20000
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trainer")


# ----------------------------------------------------------------------------
# Reproducibilidad
# ----------------------------------------------------------------------------
def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Datos: Common Voice es
# ----------------------------------------------------------------------------
class CVesDataset(Dataset):
    """Lee un TSV de Common Voice y decodifica mp3 -> waveform 16 kHz mono."""

    def __init__(
        self,
        tsv_path: str,
        clips_dir: str,
        durations_path: str | None = None,
        sample_rate: int = 16000,
        max_duration: float = 30.0,
    ):
        self.clips_dir = Path(clips_dir)
        self.sample_rate = sample_rate
        df = pd.read_csv(tsv_path, sep="\t")
        # path es el nombre del archivo mp3; sentence es la transcripción.
        df = df[["path", "sentence"]].dropna()
        df = df[df["path"].str.endswith(".mp3")]
        df = df[df["sentence"].str.strip().str.len() > 0]

        # Duraciones para bucketing/sort.
        dur_map: Dict[str, float] = {}
        if durations_path and Path(durations_path).exists():
            try:
                ddf = pd.read_csv(durations_path, sep="\t")
                for _, r in ddf.iterrows():
                    dur_map[r["clip"]] = float(r["duration[ms]"]) / 1000.0
            except Exception:
                log.warning("No se pudo leer %s, se estimará duración", durations_path)

        rows = []
        for _, r in df.iterrows():
            dur = dur_map.get(r["path"])
            if dur is not None and (dur <= 0 or dur > max_duration):
                continue
            rows.append((r["path"], str(r["sentence"]).strip(), dur))
        self.rows = rows
        self.has_durations = durations_path is not None
        log.info("Dataset %s: %d clips (max_dur=%.0fs)", tsv_path, len(rows), max_duration)

    def __len__(self):
        return len(self.rows)

    def _load_audio(self, rel: str) -> torch.Tensor:
        path = self.clips_dir / rel
        # soundfile (libsndfile >=1.1) soporta mp3 de forma nativa.
        import soundfile as sf

        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav[:, 0])  # mono
        if sr != self.sample_rate:
            wav = audio_resample(wav, sr, self.sample_rate)
        return wav

    def __getitem__(self, idx):
        rel, text, dur = self.rows[idx]
        wav = self._load_audio(rel)
        if dur is None:
            dur = wav.numel() / self.sample_rate
        return {"wav": wav, "text": text, "duration": dur, "path": rel}


def audio_resample(wav: torch.Tensor, orig: int, target: int) -> torch.Tensor:
    if orig == target:
        return wav
    try:
        import torchaudio

        return torchaudio.functional.resample(wav, orig, target)
    except Exception:
        # fallback: scipy resample_poly (lento pero sin dependencias extra).
        from scipy.signal import resample_poly

        n = int(wav.numel() * target / orig)
        out = resample_poly(wav.numpy(), target, orig)
        if len(out) > n:
            out = out[:n]
        return torch.from_numpy(out.astype("float32"))


# ----------------------------------------------------------------------------
# Fbank on-the-fly (idéntico a auden: 80 bins, dither=0, snip_edges=False)
# ----------------------------------------------------------------------------
class FbankExtractor:
    def __init__(self, sample_rate: int = 16000, num_mel_bins: int = 80):
        try:
            from lhotse.features import Fbank, FbankConfig

            cfg = FbankConfig(
                sampling_rate=sample_rate,
                num_mel_bins=num_mel_bins,
                dither=0.0,
                snip_edges=False,
            )
            self.fbank = Fbank(cfg)
            self.sample_rate = sample_rate
            self.num_mel_bins = num_mel_bins
            self.kind = "lhotse"
        except Exception:
            log.warning("lhotse no disponible; usando torchaudio kaldi-fbank")
            import torchaudio.compliance.kaldi as kaldi

            self.kaldi = kaldi
            self.kind = "kaldi"
            self.num_mel_bins = num_mel_bins

    def extract(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: 1D float32 -> (T, F) float32."""
        if self.kind == "lhotse":
            feat = self.fbank.extract(wav.numpy(), sampling_rate=self.sample_rate)
            return torch.from_numpy(np.asarray(feat)).float()
        else:
            x = wav.unsqueeze(0)  # (1, T)
            feat = self.kaldi.fbank(
                x,
                num_mel_bins=self.num_mel_bins,
                sample_frequency=self.sample_rate,
                dither=0.0,
                snip_edges=False,
            )
            return feat


# ----------------------------------------------------------------------------
# Bucketing por duración + padding dinámico
# ----------------------------------------------------------------------------
def make_batches_by_duration(
    ds: CVesDataset, batch_seconds: float, shuffle: bool = True, rng=None
) -> List[List[int]]:
    if rng is None:
        rng = np.random.default_rng()
    idxs = list(range(len(ds)))
    if shuffle:
        rng.shuffle(idxs)
    # ordenar por duración dentro de bloques grandes para batches homogéneos
    block = max(256, int(batch_seconds // 2))
    batches: List[List[int]] = []
    for s in range(0, len(idxs), block):
        chunk = idxs[s : s + block]
        chunk.sort(key=lambda i: ds.rows[i][2] or 0.0)
        cur: List[int] = []
        cur_sec = 0.0
        for i in chunk:
            d = ds.rows[i][2] or 0.0
            if cur and (cur_sec + d > batch_seconds or len(cur) >= 32):
                batches.append(cur)
                cur, cur_sec = [], 0.0
            cur.append(i)
            cur_sec += d
        if cur:
            batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(batch: List[dict], fbank: FbankExtractor, pad_val: float = math.log(1e-10)):
    feats = []
    texts = []
    for b in batch:
        f = fbank.extract(b["wav"])
        feats.append(f)
        texts.append(b["text"])
    lens = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    T = int(lens.max().item())
    F = feats[0].size(1)
    x = torch.full((len(feats), T, F), pad_val, dtype=torch.float32)
    for i, f in enumerate(feats):
        x[i, : f.size(0)] = f
    return x, lens, texts


# ----------------------------------------------------------------------------
# Optimizadores híbridos
# ----------------------------------------------------------------------------
def build_optimizers(
    model: torch.nn.Module,
    optimizer_name: str,
    lr_muon: float,
    lr_adamw: float,
    weight_decay: float,
    ns_dtype: torch.dtype,
):
    """TurboMuon sobre parámetros 2D, AdamW sobre el resto. O AdamW global."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if optimizer_name == "turbo_muon":
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        else:
            adamw_params.append(p)

    optimizers = []
    if muon_params:
        from turbo_muon import TurboMuon

        opt_m = TurboMuon(
            muon_params, lr=lr_muon, weight_decay=weight_decay, ns_dtype=ns_dtype
        )
        optimizers.append(("turbo_muon", opt_m))
        log.info(
            "TurboMuon: %d tensores 2D (%.1fM params)",
            len(muon_params),
            sum(p.numel() for p in muon_params) / 1e6,
        )
    if adamw_params:
        opt_a = torch.optim.AdamW(
            adamw_params, lr=lr_adamw, weight_decay=weight_decay
        )
        optimizers.append(("adamw", opt_a))
        log.info(
            "AdamW: %d tensores (%.1fM params)",
            len(adamw_params),
            sum(p.numel() for p in adamw_params) / 1e6,
        )
    return optimizers


def get_lrs(optimizers) -> Dict[str, float]:
    out = {}
    for name, opt in optimizers:
        if opt.param_groups:
            out[name] = opt.param_groups[0].get("lr", float("nan"))
    return out


def set_lrs(optimizers, lrs: Dict[str, float]):
    for name, opt in optimizers:
        if name in lrs:
            for g in opt.param_groups:
                g["lr"] = lrs[name]


# ----------------------------------------------------------------------------
# LR schedule: warmup lineal + coseno
# ----------------------------------------------------------------------------
class WarmupCosine:
    def __init__(self, optimizers, warmup_steps: int, total_steps: int,
                 base_lrs: Dict[str, float], min_frac: float = 0.1):
        self.optimizers = optimizers
        self.warmup = max(1, warmup_steps)
        self.total = max(1, total_steps)
        self.base = base_lrs
        self.min_frac = min_frac
        # reducción temporal de LR tras spike/rollback
        self._temp_factor = 1.0
        self._temp_left = 0
        self._temp_ramp = 1

    def trigger_temp(self, factor: float, cooldown: int, ramp: int):
        """LR *= factor durante `cooldown` steps, luego rampa lineal a 1.0
        en `ramp` steps (para rollbacks de spikes)."""
        self._temp_factor = factor
        self._temp_left = cooldown + ramp
        self._temp_ramp = max(1, ramp)

    def temp_scale(self) -> float:
        if self._temp_left <= 0:
            return 1.0
        if self._temp_left <= self._temp_ramp:
            return self._temp_factor + (1 - self._temp_factor) * (
                self._temp_ramp - self._temp_left
            ) / self._temp_ramp
        return self._temp_factor

    def advance_temp(self):
        if self._temp_left > 0:
            self._temp_left -= 1

    def scale_base(self, factor: float):
        """Multiplica los base LR de forma permanente y los aplica ya a los
        param_groups (sobrevive a los próximos step(), que recalculan desde base)."""
        for name in self.base:
            self.base[name] *= factor
        set_lrs(self.optimizers, {n: self.base[n] * self.temp_scale() for n in self.base})
        return get_lrs(self.optimizers)

    def step(self, step: int):
        ts = self.temp_scale()
        for name in self.base:
            if step < self.warmup:
                frac = step / self.warmup
            else:
                p = (step - self.warmup) / max(1, self.total - self.warmup)
                p = min(1.0, max(0.0, p))
                frac = self.min_frac + 0.5 * (1 - self.min_frac) * (1 + math.cos(math.pi * p))
            set_lrs(self.optimizers, {name: self.base[name] * frac * ts})
        return get_lrs(self.optimizers)


# ----------------------------------------------------------------------------
# Spike guard: z-score del grad norm con EMAs (estilo ZClip, arXiv:2504.02507)
# ----------------------------------------------------------------------------
class GradNormTracker:
    """EMAs de media/desviación del grad norm pre-clip. Detecta spikes por
    z-score: tier 1 (moderado) -> clip adaptativo; tier 2 (severo) -> rollback.
    Los valores spiky no contaminan las estadísticas (tier 2 excluido,
    tier 1 aportado capado a mu + z*sigma)."""

    def __init__(self, alpha: float = 0.97, warmup: int = 25,
                 z_thresh: float = 2.5, z_rollback: float = 5.0):
        self.alpha = alpha
        self.warmup = max(2, warmup)
        self.z_thresh = z_thresh
        self.z_rollback = z_rollback
        self.mu = None
        self.sigma = 0.0
        self.n = 0
        self._hist: List[float] = []

    @property
    def ready(self) -> bool:
        return self.n >= self.warmup

    def observe(self, g: float):
        """Actualiza EMAs y retorna (z, tier, cap).
        tier: 0 normal | 1 spike moderado (cap = norma objetivo del clip
        adaptativo, reciprocual z* = z_thres²/z) | 2 spike severo (rollback)."""
        if not self.ready:
            self._warmup_update(g)
            return 0.0, 0, None
        # floor relativo: evita z explosivos por sigma~0
        sigma = max(self.sigma, 1e-3 * max(1e-12, abs(self.mu)))
        z = (g - self.mu) / sigma
        if z > self.z_rollback:
            return z, 2, None
        if z > self.z_thresh:
            cap = self.mu + (self.z_thresh ** 2 / z) * sigma
            self._ema_update(min(g, self.mu + self.z_thresh * sigma))
            return z, 1, cap
        self._ema_update(g)
        return z, 0, None

    def _warmup_update(self, g: float):
        self._hist.append(g)
        self.n = len(self._hist)
        if len(self._hist) >= self.warmup:
            # init estilo ZClip: media/std muestral de la ventana de warmup
            m = sum(self._hist) / len(self._hist)
            v = sum((x - m) ** 2 for x in self._hist) / len(self._hist)
            self.mu, self.sigma = m, math.sqrt(v)
            self._hist = []

    def _ema_update(self, g: float):
        a = self.alpha
        new_mu = a * self.mu + (1 - a) * g
        self.sigma = math.sqrt(a * self.sigma ** 2 + (1 - a) * (g - self.mu) ** 2)
        self.mu = new_mu
        self.n += 1

    def state(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma, "n": self.n}

    def load(self, s: dict):
        if s and s.get("mu") is not None:
            self.mu, self.sigma, self.n = s["mu"], s["sigma"], s["n"]


def _tensors_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tensors_to_cpu(v) for v in obj)
    return obj


def _tensors_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _tensors_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tensors_to_device(v, device) for v in obj)
    return obj


def snapshot_state(model, optimizers, tracker: GradNormTracker, step: int,
                   nan_failures: int, lr_halvings: int, rollbacks: int,
                   best_wer: float) -> dict:
    """Snapshot 'last known good' en RAM CPU para rollbacks de spikes."""
    return {
        "step": step,
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizers": {name: _tensors_to_cpu(opt.state_dict()) for name, opt in optimizers},
        "tracker": tracker.state(),
        "nan_failures": nan_failures,
        "lr_halvings": lr_halvings,
        "rollbacks": rollbacks,
        "best_wer": best_wer,
    }


def restore_snapshot(snap: dict, model, optimizers, tracker: GradNormTracker, device):
    model.load_state_dict(snap["model_state"])
    for name, opt in optimizers:
        if name in snap["optimizers"]:
            opt.load_state_dict(_tensors_to_device(snap["optimizers"][name], device))
    tracker.load(snap["tracker"])


# ----------------------------------------------------------------------------
# Gradiente total + clipping + protección inf/nan
# ----------------------------------------------------------------------------
def grads_have_inf_nan(model) -> Tuple[bool, str]:
    """True si algún gradiente de un parámetro entrenable tiene inf/nan.
    Retorna también el nombre del primer parámetro ofensor (para logging)."""
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if not torch.isfinite(p.grad).all().item():
                return True, name
    return False, ""


def compute_grad_norm_and_clip(model, max_norm: float) -> Tuple[float, float]:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().item())
    total_norm = math.sqrt(total)
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm, total_norm / max(1.0, max_norm) if max_norm > 0 else 1.0


# ----------------------------------------------------------------------------
# Pérdida RNNT (receta auden: simple + pruned con warmup de escalas)
# ----------------------------------------------------------------------------
def compute_loss(outputs, step: int, warm_step: int, simple_loss_scale: float):
    simple = outputs["simple_loss"]
    pruned = outputs["pruned_loss"]
    if step >= warm_step:
        s_scale = simple_loss_scale
        p_scale = 1.0
    else:
        t = step / max(1, warm_step)
        s_scale = 1.0 - t * (1.0 - simple_loss_scale)
        p_scale = 0.1 + 0.9 * t
    loss = s_scale * simple + p_scale * pruned
    return loss, simple.detach(), pruned.detach(), s_scale, p_scale


# ----------------------------------------------------------------------------
# WER (jiwer)
# ----------------------------------------------------------------------------
def compute_wer(refs: List[str], hyps: List[str]) -> Tuple[float, float]:
    try:
        import jiwer

        wer = jiwer.wer(refs, hyps)
    except Exception:
        # fallback ingenuo por edit distance a nivel de palabra
        def ed(a, b):
            m, n = len(a), len(b)
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                prev = dp[0]
                dp[0] = i
                for j in range(1, n + 1):
                    cur = dp[j]
                    dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
                    prev = cur
            return dp[n]

        tot_err = tot = 0
        for r, h in zip(refs, hyps):
            rw, hw = r.split(), h.split()
            tot_err += ed(rw, hw)
            tot += len(rw)
        wer = tot_err / max(1, tot)
    cer = 0.0
    return float(wer), float(cer)


# ----------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizers,
    scheduler,
    step: int,
    best_wer: float,
    args,
    extra: dict | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizers": {name: opt.state_dict() for name, opt in optimizers},
        "best_wer": best_wer,
        "args": vars(args),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, str(path))
    log.info("checkpoint guardado: %s (step=%d)", path.name, step)


def save_model_export(path: Path, model: torch.nn.Module):
    """Exporta solo pesos del modelo (para inferencia)."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path / "model.pt"))
    try:
        model.config.save_pretrained(str(path))
    except Exception:
        pass


def manage_rolling(ckpt_dir: Path, keep_last: int):
    """Mantiene sólo los últimos `keep_last` checkpoints rolling."""
    rolling = sorted(
        ckpt_dir.glob("rolling_step-*.pt"),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    for old in rolling[:-keep_last]:
        old.unlink(missing_ok=True)
        log.info("rolling eliminado: %s", old.name)


def manage_best3(ckpt_dir: Path, new_path: Path, new_wer: float):
    """Mantiene los 3 mejores checkpoints por WER (menor = mejor)."""
    best_path = ckpt_dir / "best_meta.json"
    meta: List[Dict] = []
    if best_path.exists():
        meta = json.loads(best_path.read_text())
    meta.append({"file": new_path.name, "wer": new_wer})
    meta.sort(key=lambda x: x["wer"])
    to_remove = meta[3:]
    for m in to_remove:
        f = ckpt_dir / m["file"]
        f.unlink(missing_ok=True)
        log.info("best eliminado (fuera del top3): %s WER=%.4f", m["file"], m["wer"])
    meta = meta[:3]
    best_path.write_text(json.dumps(meta, indent=2))
    return meta


# ----------------------------------------------------------------------------
# Validación: loss + WER greedy
# ----------------------------------------------------------------------------
@torch.no_grad()
def validate(model, ds: CVesDataset, fbank: FbankExtractor, device,
             num_samples: int, batch_seconds: float):
    model.eval()
    n = min(num_samples, len(ds))
    rng = np.random.default_rng(123)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:n]
    # batches por duración (ordenando el subset)
    idxs.sort(key=lambda i: ds.rows[i][2] or 0.0)
    batches: List[List[int]] = []
    cur, cur_sec = [], 0.0
    for i in idxs:
        d = ds.rows[i][2] or 0.0
        if cur and (cur_sec + d > batch_seconds or len(cur) >= 16):
            batches.append(cur)
            cur, cur_sec = [], 0.0
        cur.append(i)
        cur_sec += d
    if cur:
        batches.append(cur)

    total_loss = 0.0
    total_frames = 0
    refs: List[str] = []
    hyps: List[str] = []
    for batch_idxs in batches:
        batch = [ds[j] for j in batch_idxs]
        x, lens, texts = collate(batch, fbank)
        x = x.to(device)
        lens = lens.to(device)
        try:
            outputs = model(
                x=x, x_lens=lens, source_texts=texts, target_texts=texts,
                forward_attention_decoder=False, forward_s2t_alignment=False,
                return_dict=True,
            )
            simple = outputs["simple_loss"]
            pruned = outputs["pruned_loss"]
            nf = (lens // 4).sum().item()
            total_loss += float((simple + pruned).item())
            total_frames += nf
        except Exception as e:
            log.warning("loss val falló en un batch: %s", e)
        # greedy decode para WER
        try:
            out = model.generate((x, lens), task="transcribe")
            hyps.extend(out["hypotheses"])
            refs.extend(texts)
        except Exception as e:
            log.warning("decode val falló: %s", e)

    val_loss = total_loss / max(1, total_frames)
    wer, _ = compute_wer(refs, hyps) if refs and hyps else (float("inf"), 0.0)
    model.train()
    return val_loss, wer


# ----------------------------------------------------------------------------
# CSV logger
# ----------------------------------------------------------------------------
class CSVLogger:
    def __init__(self, path: str, columns: List[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = columns
        # migración: si existe un CSV con header viejo (sin columnas nuevas),
        # se renombra y se empieza uno nuevo en lugar de corruptar filas.
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="") as f:
                old_header = f.readline().strip().split(",")
            if old_header != columns:
                migrated = self.path.with_name(
                    f"{self.path.stem}.old-{int(time.time())}{self.path.suffix}"
                )
                self.path.replace(migrated)
                log.warning("CSV con columnas viejas; se migra: %s -> %s",
                            self.path.name, migrated.name)
        self.f = open(self.path, "a", newline="", buffering=1)
        self.writer = csv.DictWriter(self.f, fieldnames=columns)
        if self.path.stat().st_size == 0:
            self.writer.writeheader()

    def log(self, row: dict):
        self.writer.writerow(row)

    def close(self):
        self.f.close()


# ----------------------------------------------------------------------------
# Texto: normalización mínima (minúsculas, quitar ¿¡)
# ----------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("¿", "").replace("¡", "")
    return " ".join(s.split())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Auden TTA ASR en español (P40)")
    p.add_argument("--model", default="AudenAI/auden-tta-m10",
                   help="repo HF o ruta local al modelo Auden TTA")
    p.add_argument("--train-tsv", default="datasets/cv_es/train.tsv")
    p.add_argument("--test-tsv", default="datasets/cv_es/test.tsv")
    p.add_argument("--clips-dir", default="datasets/cv_es/clips")
    p.add_argument("--durations", default="datasets/cv_es/clip_durations.tsv")
    p.add_argument("--output-dir", default="exp/es-asr")
    p.add_argument("--resume", default="", help="checkpoint a resumir")
    p.add_argument("--seed", type=int, default=42)
    # datos
    p.add_argument("--batch-seconds", type=float, default=120.0,
                   help="segundos de audio por batch (bucketing)")
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--num-workers", type=int, default=4)
    # entrenamiento
    p.add_argument("--optimizer", choices=["turbo_muon", "adamw"], default="turbo_muon")
    p.add_argument("--lr-muon", type=float, default=2e-3)
    p.add_argument("--lr-adamw", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="micro-batches acumulados por step del optimizador")
    p.add_argument("--max-grad-retries", type=int, default=3,
                   help="fallos inf/nan consecutivos antes del stop de emergencia")
    # spike guard (z-score sobre grad norm, estilo ZClip)
    p.add_argument("--no-spike-guard", action="store_true",
                   help="desactiva el spike guard y los snapshots de rollback")
    p.add_argument("--spike-z", type=float, default=2.5,
                   help="z-score para spike moderado (clip adaptativo)")
    p.add_argument("--spike-z-rollback", type=float, default=5.0,
                   help="z-score para spike severo (rollback + salto de ventana)")
    p.add_argument("--spike-warmup", type=int, default=25,
                   help="steps antes de activar la detección de spikes")
    p.add_argument("--spike-alpha", type=float, default=0.97,
                   help="factor EMA de las estadísticas del grad norm")
    p.add_argument("--spike-lr-mode", choices=["temporary", "permanent", "none"],
                   default="temporary", help="reducción de LR tras un rollback")
    p.add_argument("--spike-lr-factor", type=float, default=0.5,
                   help="factor de reducción temporal de LR tras rollback")
    p.add_argument("--spike-cooldown", type=int, default=300,
                   help="steps con LR reducido tras rollback (modo temporary)")
    p.add_argument("--spike-lr-ramp", type=int, default=100,
                   help="steps de rampa lineal de regreso al LR del schedule")
    p.add_argument("--max-rollbacks", type=int, default=5,
                   help="rollbacks máximos antes del stop de emergencia")
    p.add_argument("--snapshot-every", type=int, default=10,
                   help="frecuencia (steps) del snapshot 'last known good' en RAM")
    p.add_argument("--rnnt-warm-step", type=int, default=2000)
    p.add_argument("--simple-loss-scale", type=float, default=0.5)
    p.add_argument("--prune-range", type=int, default=5)
    p.add_argument("--am-scale", type=float, default=0.0)
    p.add_argument("--lm-scale", type=float, default=0.25)
    # checkpoints
    p.add_argument("--rolling-every", type=int, default=500)
    p.add_argument("--rolling-keep", type=int, default=3)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-samples", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    # wandb
    p.add_argument("--wandb", action="store_true", help="activar sincronización wandb")
    p.add_argument("--wandb-project", default="auden-asr-es")
    p.add_argument("--wandb-run-name", default="")
    # congelado
    p.add_argument("--no-freeze-branches", action="store_true",
                   help="no congelar text_encoder/attention_decoder/align")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s | sm_%d%d",
                 torch.cuda.get_device_name(0), *torch.cuda.get_device_capability(0))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- wandb ----
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config=vars(args),
            dir=str(out_dir),
        )
    else:
        os.environ.setdefault("WANDB_MODE", "disabled")

    # ---- modelo ----
    log.info("Cargando modelo %s ...", args.model)
    from auden.auto.auto_model import AutoModel

    model = AutoModel.from_pretrained(args.model, map_location="cpu")
    model = model.to(device)
    model.train()

    # ---- congelar ramas no-ASR ----
    if not args.no_freeze_branches:
        for branch in ["text_encoder", "attention_decoder"]:
            mod = getattr(model, branch, None)
            if mod is not None:
                for p_ in mod.parameters():
                    p_.requires_grad = False
                log.info("Congelado: %s", branch)
        # heads de align (parámetros sueltos)
        for nm, p_ in model.named_parameters():
            if "align" in nm or "s2t" in nm:
                p_.requires_grad = False
        trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
        total = sum(p_.numel() for p_ in model.parameters())
        log.info("Params: %s/%s (%.1f%%)", f"{trainable/1e6:.1f}M", f"{total/1e6:.1f}M",
                 100 * trainable / max(1, total))

    # ---- dtype NS: fp32 en P40 (sm<80) ----
    ns_dtype = torch.float32
    if device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
        ns_dtype = torch.bfloat16
        log.info("NS dtype: bfloat16 (GPU sm>=80)")
    else:
        log.info("NS dtype: float32 (P40 / sm<80)")

    # ---- optimizadores ----
    optimizers = build_optimizers(
        model, args.optimizer, args.lr_muon, args.lr_adamw,
        args.weight_decay, ns_dtype,
    )
    base_lrs = get_lrs(optimizers)
    scheduler = WarmupCosine(optimizers, args.warmup_steps, args.max_steps, base_lrs)

    # ---- spike guard ----
    spike_guard = not args.no_spike_guard
    tracker = GradNormTracker(args.spike_alpha, args.spike_warmup,
                              args.spike_z, args.spike_z_rollback)
    snap = None
    rollbacks = 0
    lr_halvings = 0
    nan_failures = 0

    def _extra_state() -> dict:
        return {"lr_halvings": lr_halvings, "rollbacks": rollbacks,
                "grad_tracker": tracker.state()}

    # ---- feature extractor ----
    fbank = FbankExtractor()

    # ---- datasets ----
    train_ds = CVesDataset(
        args.train_tsv, args.clips_dir, args.durations,
        16000, args.max_duration,
    )
    val_ds = CVesDataset(
        args.test_tsv, args.clips_dir, args.durations,
        16000, args.max_duration,
    )

    # ---- resume ----
    start_step = 0
    best_wer = float("inf")

    def _restore_ckpt_state(ckpt: dict):
        nonlocal lr_halvings, rollbacks
        if ckpt.get("grad_tracker"):
            tracker.load(ckpt["grad_tracker"])
        if ckpt.get("lr_halvings"):
            # los halvings permanentes viven en scheduler.base, que no se
            # guarda: se reaplican al reanudar
            lr_halvings = ckpt["lr_halvings"]
            scheduler.scale_base(0.5 ** lr_halvings)
        rollbacks = ckpt.get("rollbacks", 0)

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=False)
        for name, opt in optimizers:
            if name in ckpt["optimizers"]:
                opt.load_state_dict(ckpt["optimizers"][name])
        start_step = ckpt["step"]
        best_wer = ckpt.get("best_wer", float("inf"))
        _restore_ckpt_state(ckpt)
        log.info("Resumido desde step %d (best_wer=%.4f)", start_step, best_wer)
    elif not args.resume:
        # auto-resume desde el rolling más reciente
        rolling = sorted(ckpt_dir.glob("rolling_step-*.pt"),
                         key=lambda p: int(p.stem.split("-")[1]))
        if rolling:
            args.resume = str(rolling[-1])
            ckpt = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(ckpt["model_state"], strict=False)
            for name, opt in optimizers:
                if name in ckpt["optimizers"]:
                    opt.load_state_dict(ckpt["optimizers"][name])
            start_step = ckpt["step"]
            best_wer = ckpt.get("best_wer", float("inf"))
            _restore_ckpt_state(ckpt)
            log.info("Auto-resume desde %s (step %d)", rolling[-1].name, start_step)

    # ---- loggers CSV ----
    train_log = CSVLogger(str(out_dir / "train_log.csv"), [
        "step", "epoch", "loss", "simple_loss", "pruned_loss",
        "s_scale", "p_scale", "lr_muon", "lr_adamw",
        "grad_norm", "clip_ratio", "frames", "batch_size",
        "tok_per_sec", "batches_per_sec", "mem_alloc_gb", "mem_reserved_gb",
        "elapsed_sec", "nan_failures", "lr_scale",
        "gn_z", "gn_mu", "gn_sigma", "spike_tier", "zclip_cap",
        "rollbacks", "lr_temp_scale",
    ])
    val_log = CSVLogger(str(out_dir / "val_log.csv"), [
        "step", "val_loss", "wer",
    ])

    # ---- bucle ----
    accum = max(1, args.grad_accum_steps)

    rng = np.random.default_rng(args.seed)
    epoch = 0
    step = start_step
    t0 = time.time()
    last_log = time.time()
    last_frames = 0

    log.info("Iniciando entrenamiento: %d steps, batch~%.0fs, grad_accum=%d",
             args.max_steps, args.batch_seconds, accum)

    while step < args.max_steps:
        epoch += 1
        batches = make_batches_by_duration(train_ds, args.batch_seconds, shuffle=True, rng=rng)
        # ventanas de `accum` micro-batches (la última puede quedar corta)
        windows = [batches[i : i + accum] for i in range(0, len(batches), accum)]
        for window in windows:
            if step >= args.max_steps:
                break

            # ---- collate de los micro-batches (una sola vez por ventana) ----
            micros = []
            skip = False
            for batch_idx in window:
                batch = [train_ds[j] for j in batch_idx]
                try:
                    x, lens, texts = collate(batch, fbank)
                except Exception as e:
                    log.warning("collate falló: %s", e)
                    skip = True
                    break
                micros.append((x, lens, texts, len(batch_idx)))
            if skip or not micros:
                continue
            micros = [
                (x.to(device, non_blocking=True),
                 lens.to(device, non_blocking=True),
                 [normalize_text(t) for t in texts], bs)
                for x, lens, texts, bs in micros
            ]

            # ---- intentos con protección inf/nan (misma ventana hasta éxito) ----
            skip_window = False
            while True:
                for opt in optimizers:
                    opt[1].zero_grad(set_to_none=True)

                bad_loss = False
                loss_sum = simple_sum = pruned_sum = 0.0
                nf_total = bs_total = 0
                for x, lens, texts, bs in micros:
                    outputs = model(
                        x=x, x_lens=lens, source_texts=texts, target_texts=texts,
                        prune_range=args.prune_range, am_scale=args.am_scale,
                        lm_scale=args.lm_scale,
                        forward_attention_decoder=False, forward_s2t_alignment=False,
                        return_dict=True,
                    )
                    loss, simple, pruned, s_scale, p_scale = compute_loss(
                        outputs, step, args.rnnt_warm_step, args.simple_loss_scale
                    )
                    nf = (lens // 4).sum().item()
                    if not torch.isfinite(loss).all().item():
                        bad_loss = True
                        log.warning("step %d: loss no finita (%.4g), ventana descartada",
                                    step + 1, float(loss.item()))
                        break
                    # normalización por frame + escalado por acumulación
                    (loss / max(1, nf) / len(micros)).backward()
                    loss_sum += float(loss.item()) / max(1, nf)
                    simple_sum += float(simple.item()) / max(1, nf)
                    pruned_sum += float(pruned.item()) / max(1, nf)
                    nf_total += nf
                    bs_total += bs

                if bad_loss:
                    grad_bad, bad_name = True, "loss"
                else:
                    grad_bad, bad_name = grads_have_inf_nan(model)

                if grad_bad:
                    # descartar grads contaminados, halving permanente, reintentar
                    for opt in optimizers:
                        opt[1].zero_grad(set_to_none=True)
                    nan_failures += 1
                    lr_halvings += 1
                    lrs = scheduler.scale_base(0.5)
                    log.warning(
                        "inf/nan en '%s' | fallo consecutivo %d/%d | lr_scale %.4f | lrs %s",
                        bad_name, nan_failures, args.max_grad_retries,
                        0.5 ** lr_halvings, lrs,
                    )
                    if nan_failures >= args.max_grad_retries:
                        em_path = ckpt_dir / f"emergency_step-{step:07d}.pt"
                        save_checkpoint(
                            em_path, model, optimizers, scheduler, step, best_wer, args,
                            extra={**_extra_state(), "emergency": True,
                                   "nan_failures": nan_failures},
                        )
                        train_log.close()
                        val_log.close()
                        if wandb_run:
                            wandb_run.finish(exit_code=1)
                        log.error("%d fallos consecutivos con inf/nan. "
                                  "Stop de emergencia: %s", nan_failures, em_path)
                        sys.exit(1)
                    continue  # reintentar la misma ventana con LR reducido

                # ---- update (ventana limpia) ----
                grad_norm, clip_ratio = compute_grad_norm_and_clip(model, args.grad_clip)

                # ---- spike guard: z-score del grad norm pre-clip ----
                spike_tier, gn_z, gn_mu, gn_sigma, zclip_cap = 0, 0.0, 0.0, 0.0, -1.0
                if spike_guard:
                    gn_z, spike_tier, cap = tracker.observe(grad_norm)
                    gn_mu = tracker.mu if tracker.mu is not None else 0.0
                    gn_sigma = tracker.sigma
                    if spike_tier == 1 and cap is not None:
                        zclip_cap = cap
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cap)
                        clip_ratio = grad_norm / max(1.0, cap)
                        log.warning("spike tier1: gn %.2f -> cap %.2f (z=%.1f)",
                                    grad_norm, cap, gn_z)
                    elif spike_tier == 2:
                        rollbacks += 1
                        log.warning("spike tier2: gn %.2f (z=%.1f) | rollback %d/%d",
                                    grad_norm, gn_z, rollbacks, args.max_rollbacks)
                        if rollbacks > args.max_rollbacks:
                            em_path = ckpt_dir / f"emergency_step-{step:07d}.pt"
                            save_checkpoint(
                                em_path, model, optimizers, scheduler, step, best_wer,
                                args, extra={**_extra_state(), "emergency": True,
                                             "reason": "spike_rollbacks"},
                            )
                            train_log.close()
                            val_log.close()
                            if wandb_run:
                                wandb_run.finish(exit_code=1)
                            log.error("%d rollbacks por spikes (máx %d). "
                                      "Stop de emergencia: %s",
                                      rollbacks, args.max_rollbacks, em_path)
                            sys.exit(1)
                        # rollback al último snapshot sano y salto de ventana
                        if snap is None:
                            log.warning("sin snapshot todavía: ventana saltada sin update")
                        else:
                            restore_snapshot(snap, model, optimizers, tracker, device)
                            step = snap["step"]
                            nan_failures = snap["nan_failures"]
                            lr_halvings = snap["lr_halvings"]
                            best_wer = snap["best_wer"]
                            log.info("rollback a step %d (snapshot sano)", step)
                        if args.spike_lr_mode == "temporary":
                            scheduler.trigger_temp(args.spike_lr_factor,
                                                   args.spike_cooldown,
                                                   args.spike_lr_ramp)
                        elif args.spike_lr_mode == "permanent":
                            scheduler.scale_base(args.spike_lr_factor)
                            lr_halvings += 1
                        skip_window = True
                        break

                lrs = scheduler.step(step)
                for _, opt in optimizers:
                    opt.step()
                scheduler.advance_temp()

                nan_failures = 0
                step += 1
                n_micros = len(micros)
                win_loss = loss_sum / n_micros
                win_simple = simple_sum / n_micros
                win_pruned = pruned_sum / n_micros
                now = time.time()
                dt = now - last_log
                if step % args.log_every == 0 or step == 1:
                    mem_a = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
                    mem_r = torch.cuda.memory_reserved() / 1e9 if device.type == "cuda" else 0
                    fps = (step - start_step) / (now - t0)
                    row = {
                        "step": step, "epoch": epoch,
                        "loss": win_loss, "simple_loss": win_simple,
                        "pruned_loss": win_pruned,
                        "s_scale": s_scale, "p_scale": p_scale,
                        "lr_muon": lrs.get("turbo_muon", float("nan")),
                        "lr_adamw": lrs.get("adamw", float("nan")),
                        "grad_norm": grad_norm, "clip_ratio": clip_ratio,
                        "frames": nf_total, "batch_size": bs_total,
                        "tok_per_sec": 0.0,
                        "batches_per_sec": fps,
                        "mem_alloc_gb": mem_a, "mem_reserved_gb": mem_r,
                        "elapsed_sec": now - t0,
                        "nan_failures": nan_failures,
                        "lr_scale": 0.5 ** lr_halvings,
                        "gn_z": gn_z, "gn_mu": gn_mu, "gn_sigma": gn_sigma,
                        "spike_tier": spike_tier, "zclip_cap": zclip_cap,
                        "rollbacks": rollbacks,
                        "lr_temp_scale": scheduler.temp_scale(),
                    }
                    train_log.log(row)
                    if wandb_run:
                        wandb_run.log(row, step=step)
                    log.info(
                        "step %d | loss %.4f | lr_m %.5f lr_a %.5f | gn %.2f (z %.1f) | %.0f batches/s | %.1fGB",
                        step, win_loss,
                        row["lr_muon"], row["lr_adamw"], grad_norm, gn_z, fps, mem_a,
                    )
                    last_log = now
                    last_frames = nf_total
                # ---- snapshot 'last known good' para rollbacks ----
                if spike_guard and step % args.snapshot_every == 0:
                    snap = snapshot_state(model, optimizers, tracker, step,
                                          nan_failures, lr_halvings, rollbacks, best_wer)
                break  # ventana completada con éxito

            if skip_window:
                continue  # saltar rolling/validación para la ventana descartada

            # ---- checkpoint rolling ----
            if step % args.rolling_every == 0:
                rp = ckpt_dir / f"rolling_step-{step:07d}.pt"
                save_checkpoint(rp, model, optimizers, scheduler, step, best_wer,
                                args, extra=_extra_state())
                manage_rolling(ckpt_dir, args.rolling_keep)

            # ---- validación + best ----
            if step % args.val_every == 0:
                vloss, wer = validate(
                    model, val_ds, fbank, device, args.val_samples, args.batch_seconds
                )
                val_log.log({"step": step, "val_loss": vloss, "wer": wer})
                if wandb_run:
                    wandb_run.log({"val_loss": vloss, "wer": wer}, step=step)
                log.info("VALID step %d | val_loss %.4f | WER %.4f", step, vloss, wer)
                if wer < best_wer:
                    best_wer = wer
                    bp = ckpt_dir / f"best_step-{step:07d}_wer-{wer:.4f}.pt"
                    save_checkpoint(bp, model, optimizers, scheduler, step, best_wer,
                                    args, extra=_extra_state())
                    save_model_export(out_dir / "best_model", model)
                    manage_best3(ckpt_dir, bp, wer)
                gc.collect()
                torch.cuda.empty_cache()

    # ---- fin ----
    final_path = ckpt_dir / f"final_step-{step:07d}.pt"
    save_checkpoint(final_path, model, optimizers, scheduler, step, best_wer,
                    args, extra=_extra_state())
    save_model_export(out_dir / "final_model", model)
    train_log.close()
    val_log.close()
    if wandb_run:
        wandb_run.finish()
    log.info("Entrenamiento finalizado. step=%d best_wer=%.4f", step, best_wer)


if __name__ == "__main__":
    main()