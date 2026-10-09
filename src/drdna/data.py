"""CIFAR-10 loading and the leakage-free splits used by the evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchvision
from torchvision import transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)


@dataclass
class Splits:
    """Disjoint image sets (normalized tensors) and their labels.

    Attributes:
        profile: Training images used to build the tau1/tau2/tau3 profiles.
        calib: Held-out clean test images used only to choose the detection threshold.
        evaluation: Further held-out clean test images used for fault trials and for
            measuring the false-positive rate.
    """

    profile: torch.Tensor
    calib: torch.Tensor
    evaluation: torch.Tensor
    evaluation_labels: torch.Tensor


def _load(
    root: str, train: bool, idx: np.ndarray, download: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=tfm)
    xs = torch.stack([ds[int(i)][0] for i in idx])
    ys = torch.tensor([ds.targets[int(i)] for i in idx])
    return xs, ys


def make_splits(
    root: str,
    n_profile: int = 1000,
    n_calib: int = 2000,
    n_eval: int = 2000,
    seed: int = 0,
    download: bool = False,
) -> Splits:
    """Draw disjoint profile / calibration / evaluation sets.

    Profiling uses *training* images; calibration and evaluation use disjoint slices of the
    *test* set, so the threshold is never tuned on the images it is evaluated on.

    Args:
        root: Directory containing ``cifar-10-batches-py``.
        n_profile: Number of training images for profiling.
        n_calib: Number of test images for threshold calibration.
        n_eval: Number of test images for evaluation.
        seed: Seed for the random draw.
        download: Download CIFAR-10 if missing.
    """
    rng = np.random.default_rng(seed)
    tr = rng.permutation(50_000)[:n_profile]
    te = rng.permutation(10_000)
    if n_calib + n_eval > len(te):
        raise ValueError("calibration + evaluation exceed the 10000 test images")
    px, _ = _load(root, True, tr, download)
    cx, _ = _load(root, False, te[:n_calib], download)
    ex, ey = _load(root, False, te[n_calib : n_calib + n_eval], download)
    return Splits(px, cx, ex, ey)
