#!/usr/bin/env python
"""Fine-tuning ASR en español sobre Auden TTA (AudenAI/auden-tta-m10) con
Common Voice es, optimizado para Tesla P40 (sm_61, 24 GB, fp32).

Características:
  - Optimizador Turbo-Muon (híbrido con AdamW) o AdamW puro (--optimizer).
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

    def step(self, step: int):
        for name in self.base:
            if step < self.warmup:
                frac = step / self.warmup
            else:
                p = (step - self.warmup) / max(1, self.total - self.warmup)
                p = min(1.0, max(0.0, p))
                frac = self.min_frac + 0.5 * (1 - self.min_frac) * (1 + math.cos(math.pi * p))
            set_lrs(self.optimizers, {name: self.base[name] * frac})
        return get_lrs(self.optimizers)


# ----------------------------------------------------------------------------
# Gradiente total + clipping
# ----------------------------------------------------------------------------
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
):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizers": {name: opt.state_dict() for name, opt in optimizers},
        "best_wer": best_wer,
        "args": vars(args),
    }
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
    p.add_argument("--lr-muon", type=float, default=0.02)
    p.add_argument("--lr-adamw", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--grad-clip", type=float, default=5.0)
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
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=False)
        for name, opt in optimizers:
            if name in ckpt["optimizers"]:
                opt.load_state_dict(ckpt["optimizers"][name])
        start_step = ckpt["step"]
        best_wer = ckpt.get("best_wer", float("inf"))
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
            log.info("Auto-resume desde %s (step %d)", rolling[-1].name, start_step)

    # ---- loggers CSV ----
    train_log = CSVLogger(str(out_dir / "train_log.csv"), [
        "step", "epoch", "loss", "simple_loss", "pruned_loss",
        "s_scale", "p_scale", "lr_muon", "lr_adamw",
        "grad_norm", "clip_ratio", "frames", "batch_size",
        "tok_per_sec", "batches_per_sec", "mem_alloc_gb", "mem_reserved_gb",
        "elapsed_sec",
    ])
    val_log = CSVLogger(str(out_dir / "val_log.csv"), [
        "step", "val_loss", "wer",
    ])

    # ---- bucle ----
    rng = np.random.default_rng(args.seed)
    epoch = 0
    step = start_step
    t0 = time.time()
    last_log = time.time()
    last_frames = 0

    log.info("Iniciando entrenamiento: %d steps, batch~%.0fs", args.max_steps, args.batch_seconds)

    while step < args.max_steps:
        epoch += 1
        batches = make_batches_by_duration(train_ds, args.batch_seconds, shuffle=True, rng=rng)
        for batch_idx in batches:
            if step >= args.max_steps:
                break
            batch = [train_ds[j] for j in batch_idx]
            try:
                x, lens, texts = collate(batch, fbank)
            except Exception as e:
                log.warning("collate falló: %s", e)
                continue
            x = x.to(device, non_blocking=True)
            lens = lens.to(device, non_blocking=True)
            texts = [normalize_text(t) for t in texts]

            # forward + loss
            for opt in optimizers:
                opt[1].zero_grad(set_to_none=True)
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
            loss = loss / max(1, nf)  # normalización por frame para backprop estable

            loss.backward()

            grad_norm, clip_ratio = compute_grad_norm_and_clip(model, args.grad_clip)
            lrs = scheduler.step(step)
            for _, opt in optimizers:
                opt.step()

            step += 1
            now = time.time()
            dt = now - last_log
            if step % args.log_every == 0 or step == 1:
                mem_a = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
                mem_r = torch.cuda.memory_reserved() / 1e9 if device.type == "cuda" else 0
                fps = (step - start_step) / (now - t0)
                row = {
                    "step": step, "epoch": epoch,
                    "loss": float(loss.item()), "simple_loss": float(simple.item()) / max(1, nf),
                    "pruned_loss": float(pruned.item()) / max(1, nf),
                    "s_scale": s_scale, "p_scale": p_scale,
                    "lr_muon": lrs.get("turbo_muon", float("nan")),
                    "lr_adamw": lrs.get("adamw", float("nan")),
                    "grad_norm": grad_norm, "clip_ratio": clip_ratio,
                    "frames": nf, "batch_size": len(batch_idx),
                    "tok_per_sec": 0.0,
                    "batches_per_sec": fps,
                    "mem_alloc_gb": mem_a, "mem_reserved_gb": mem_r,
                    "elapsed_sec": now - t0,
                }
                train_log.log(row)
                if wandb_run:
                    wandb_run.log(row, step=step)
                log.info(
                    "step %d | loss %.4f | lr_m %.5f lr_a %.5f | gn %.2f | %.0f batches/s | %.1fGB",
                    step, float(loss.item()),
                    row["lr_muon"], row["lr_adamw"], grad_norm, fps, mem_a,
                )
                last_log = now
                last_frames = nf

            # ---- checkpoint rolling ----
            if step % args.rolling_every == 0:
                rp = ckpt_dir / f"rolling_step-{step:07d}.pt"
                save_checkpoint(rp, model, optimizers, scheduler, step, best_wer, args)
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
                    save_checkpoint(bp, model, optimizers, scheduler, step, best_wer, args)
                    save_model_export(out_dir / "best_model", model)
                    manage_best3(ckpt_dir, bp, wer)
                gc.collect()
                torch.cuda.empty_cache()

    # ---- fin ----
    final_path = ckpt_dir / f"final_step-{step:07d}.pt"
    save_checkpoint(final_path, model, optimizers, scheduler, step, best_wer, args)
    save_model_export(out_dir / "final_model", model)
    train_log.close()
    val_log.close()
    if wandb_run:
        wandb_run.finish()
    log.info("Entrenamiento finalizado. step=%d best_wer=%.4f", step, best_wer)


if __name__ == "__main__":
    main()