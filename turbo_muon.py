"""Turbo-Muon optimizer with AOL-preconditioned orthogonalization.

Adaptación para Tesla P40 (sm_61, sin GEMM bf16) de la implementación original
de erickfmm/frankenstein-transformer. Las iteraciones de Newton-Schulz se
ejecutan en float32 en lugar de bfloat16 (la P40 no acelera bf16).

Turbo-Muon (Boissin et al. 2025, arXiv:2512.04632) extiende Muon aplicando un
precondicionador AOL (Approximate Orthogonalization via Lowdin) antes de las
iteraciones de Newton-Schulz, reduciendo el número de iteraciones de 5 a 4.

Uso recomendado (híbrido): TurboMuon sobre parámetros 2D (matrices ocultas) +
AdamW sobre el resto (1D, 3D+ convoluciones, embeddings). Ver
`build_optimizers` en trainer.py.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class TurboMuon(Optimizer):
    """Turbo-Muon optimizer with AOL-preconditioned orthogonalization.

    Only parameters with ``ndim >= 2`` are processed; parámetros de menor
    dimensión se omiten (deben ir a un AdamW de respaldo).

    Args:
        params: iterable de parámetros o grupos.
        lr: learning rate (default 0.02).
        momentum: coeficiente de momentum (default 0.95).
        nesterov: momentum estilo Nesterov (default True).
        ns_steps: iteraciones de Newton-Schulz (default 4).
        ns_eps: epsilon de estabilidad (default 1e-7).
        weight_decay: weight decay desacoplado (default 0.0).
        ns_dtype: dtype interno de las iteraciones NS. Por defecto float32
            (la P40 no acelera bf16). En GPUs sm>=80 se puede usar bfloat16.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 4,
        ns_eps: float = 1e-7,
        weight_decay: float = 0.0,
        ns_dtype: torch.dtype = torch.float32,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            weight_decay=weight_decay,
            ns_dtype=ns_dtype,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.ndim < 2:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(grad)
                g = grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                a, b, c = 3.4445, -4.7750, 2.0315
                ns_dtype = group["ns_dtype"]
                x = g.to(ns_dtype)
                transposed = False
                if x.size(0) > x.size(1):
                    x = x.T
                    transposed = True

                a_mat = x @ x.T
                s = (
                    torch.sum(torch.abs(a_mat), dim=1, keepdim=True)
                    .clamp(min=group["ns_eps"])
                    .pow(-0.5)
                )
                x = x * s
                a_mat = a_mat * s * s.T

                for i in range(int(group["ns_steps"])):
                    b_mat = b * a_mat + c * (a_mat @ a_mat)
                    x = a * x + b_mat @ x
                    if i < int(group["ns_steps"]) - 1:
                        a_mat = x @ x.T

                if transposed:
                    x = x.T
                p.add_(x.to(p.dtype), alpha=-group["lr"])

        return loss