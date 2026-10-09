"""Vectorized Dr. DNA profiling and scoring (tau1, tau2, tau3).

Following the paper, each monitored layer has a *cohort* of ``k`` detection sites.

* tau1 - per-site activation histogram; a site scores ``1 - frequency(bin)`` and 1 when the
  value is outside the profiled range or non-finite.
* tau2 - layer histogram over the cohort; scored with the Earth Mover's Distance between the
  inference histogram and the profiled one.
* tau3 - location of the strongest and weakest activation of the layer; 0 when both match
  the profile, 0.5 when one does, 1 when neither does.

The layer score accumulates over depth:
``Score[l] = l1*tau1[l] + l2*tau2[l] + l3*tau3[l] + Score[l-1]``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Scores:
    """Per-layer scores for a batch (each tensor is ``[batch, layers]``).

    Attributes:
        tau1: Summed per-site abnormality (``0..k``).
        tau2: Normalized EMD between inference and profiled layer histograms (``0..1``).
        tau3: Extreme-neuron location mismatch (``0, 0.5`` or ``1``).
        layers: Layer names in execution order.
    """

    tau1: torch.Tensor
    tau2: torch.Tensor
    tau3: torch.Tensor
    layers: list[str]

    def cumulative(
        self, lambdas: tuple[float, float, float] = (1.0, 1.0, 1.0), mask: str = "123"
    ) -> torch.Tensor:
        """Depth-accumulated score ``[batch, layers]``.

        Args:
            lambdas: Weights for tau1, tau2, tau3.
            mask: Which components to use, e.g. ``"1"`` for tau1 only (ablation).
        """
        parts = {"1": self.tau1, "2": self.tau2, "3": self.tau3}
        per_layer = sum(lam * parts[m] for lam, m in zip(lambdas, "123", strict=True) if m in mask)
        return torch.cumsum(per_layer, dim=1)


def default_layers(model: nn.Module) -> list[str]:
    """Layers monitored by the reference reproduction for a CIFAR ResNet-18."""
    names = ["conv1", "bn1"]
    for stage in range(1, 5):
        names += [f"layer{stage}.0.{n}" for n in ("conv1", "bn1", "conv2", "bn2")]
    names.append("avgpool")
    modules = dict(model.named_modules())
    return [n for n in names if n in modules]


class DrDNA:
    """Dr. DNA detector.

    Args:
        layers: Monitored module names in execution order.
        cohort: Detection sites per layer (``k``).
        bins: Histogram bins.
        seed: Seed for choosing detection sites.
    """

    def __init__(
        self, layers: Sequence[str], cohort: int = 32, bins: int = 10, seed: int = 0
    ) -> None:
        self.layers = list(layers)
        self.cohort = cohort
        self.bins = bins
        self.seed = seed
        self.fitted = False
        self._sites: dict[str, torch.Tensor] = {}
        self._shape: dict[str, tuple[int, ...]] = {}
        self._lo1: dict[str, torch.Tensor] = {}
        self._hi1: dict[str, torch.Tensor] = {}
        self._f1: dict[str, torch.Tensor] = {}
        self._lo2: dict[str, float] = {}
        self._hi2: dict[str, float] = {}
        self._f2: dict[str, torch.Tensor] = {}
        self._ext: dict[str, tuple[int, int]] = {}
        self._buf: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._mode = "idle"
        self._sum: dict[str, torch.Tensor] = {}
        self._vals: dict[str, list[torch.Tensor]] = {}

    # ------------------------------------------------------------------ hooks
    def _hook(self, name: str):
        def fn(_m: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            flat = out.detach().flatten(1)
            if name not in self._sites:
                g = torch.Generator().manual_seed(self.seed + len(self._sites))
                k = min(self.cohort, flat.shape[1])
                self._sites[name] = torch.randperm(flat.shape[1], generator=g)[:k]
                self._shape[name] = tuple(out.shape[1:])
            vals = flat[:, self._sites[name]]
            if self._mode == "fit":
                finite = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
                self._sum[name] = self._sum.get(name, 0) + finite.sum(0)
                self._vals.setdefault(name, []).append(vals.clone())
            elif self._mode == "score":
                self._buf[name] = (vals.clone(), flat.argmax(1), flat.argmin(1))

        return fn

    @contextmanager
    def attach(self, model: nn.Module) -> Iterator[None]:
        """Register observation hooks on the monitored layers; removed on exit."""
        modules = dict(model.named_modules())
        handles = [modules[n].register_forward_hook(self._hook(n)) for n in self.layers]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    # ---------------------------------------------------------------- profile
    @torch.no_grad()
    def fit(self, model: nn.Module, batches: Iterator[torch.Tensor]) -> DrDNA:
        """Profile fault-free behaviour (tau1, tau2, tau3) over ``batches`` of inputs."""
        model.eval()
        self._sites.clear()
        self._sum.clear()
        self._vals.clear()
        self._mode = "fit"
        n = 0
        try:
            with self.attach(model):
                for x in batches:
                    model(x)
                    n += x.shape[0]
        finally:
            self._mode = "idle"
        if n == 0:
            raise ValueError("no profiling data")
        for name in self.layers:
            v = torch.cat(self._vals[name])  # [N, k]
            lo, hi = v.min(0).values, v.max(0).values
            hi = torch.where(hi > lo, hi, lo + 1e-6)
            self._lo1[name], self._hi1[name] = lo, hi
            self._f1[name] = self._hist(v, lo, hi) / v.shape[0]
            pooled = v.flatten()
            lo2, hi2 = float(pooled.min()), float(pooled.max())
            hi2 = hi2 if hi2 > lo2 else lo2 + 1e-6
            self._lo2[name], self._hi2[name] = lo2, hi2
            idx = self._bin(pooled, torch.tensor(lo2), torch.tensor(hi2))
            self._f2[name] = torch.bincount(idx, minlength=self.bins).float() / pooled.numel()
            mean_act = self._sum[name] / n
            self._ext[name] = (int(mean_act.argmax()), int(mean_act.argmin()))
        self.fitted = True
        return self

    def _bin(self, v: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        v = torch.nan_to_num(v, nan=float("inf"), posinf=float("inf"), neginf=float("-inf"))
        frac = (v - lo) / (hi - lo)
        return (frac * self.bins).floor().clamp(0, self.bins - 1).long()

    def _hist(self, v: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Per-column histogram counts, ``[k, bins]``."""
        idx = self._bin(v, lo, hi)  # [N, k]
        out = torch.zeros(v.shape[1], self.bins)
        out.scatter_add_(1, idx.T, torch.ones_like(idx.T, dtype=torch.float32))
        return out

    # ------------------------------------------------------------------ score
    @torch.no_grad()
    def score(self, model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, Scores]:
        """Run ``model`` on ``x`` and score the batch.

        Call inside any fault-injection context so that the faulty values are observed.

        Returns:
            ``(logits, scores)``.
        """
        if not self.fitted:
            raise RuntimeError("call fit() first")
        model.eval()
        self._buf.clear()
        self._mode = "score"
        try:
            with self.attach(model):
                logits = model(x)
        finally:
            self._mode = "idle"
        t1, t2, t3 = [], [], []
        for name in self.layers:
            vals, amax, amin = self._buf[name]
            lo, hi = self._lo1[name], self._hi1[name]
            finite = torch.isfinite(vals)
            inside = finite & (vals >= lo) & (vals <= hi)
            idx = self._bin(vals, lo, hi)
            freq = self._f1[name].gather(1, idx.T).T
            t1.append(torch.where(inside, 1.0 - freq, torch.ones_like(freq)).sum(1))
            idx2 = self._bin(vals, torch.tensor(self._lo2[name]), torch.tensor(self._hi2[name]))
            counts = torch.zeros(vals.shape[0], self.bins)
            counts.scatter_add_(1, idx2, torch.ones_like(idx2, dtype=torch.float32))
            cdf_diff = torch.cumsum(counts / vals.shape[1] - self._f2[name], dim=1)
            t2.append(cdf_diff.abs().sum(1) / (self.bins - 1))
            pmax, pmin = self._ext[name]
            t3.append(0.5 * (amax != pmax).float() + 0.5 * (amin != pmin).float())
        scores = Scores(
            torch.stack(t1, 1), torch.stack(t2, 1), torch.stack(t3, 1), list(self.layers)
        )
        return logits, scores
