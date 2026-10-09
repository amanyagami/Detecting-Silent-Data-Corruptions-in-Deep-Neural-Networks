"""CIFAR-10 ResNet-18 used by the reproduction."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from torch import nn


def cifar_resnet18(weights: str | Path | None = None) -> nn.Module:
    """Build the CIFAR-style ResNet-18 (3x3 stride-1 stem, 10 classes).

    The architecture matches ``src/models/resnet.py`` so the repository's checkpoint
    loads without key changes.

    Args:
        weights: Optional path to a state dict (``state_dicts/resnet18.pt``; fetch it with
            ``git lfs pull``).

    Returns:
        The model in eval mode.
    """
    net = torchvision.models.resnet18(num_classes=10)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if weights is not None:
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return net.eval()
