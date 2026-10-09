"""Small statistics helpers (no scipy dependency)."""

from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auroc(negatives: np.ndarray, positives: np.ndarray) -> float:
    """Area under the ROC curve (ties count half); ``positives`` should score higher."""
    neg = np.sort(np.asarray(negatives, dtype=np.float64))
    pos = np.asarray(positives, dtype=np.float64)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    return float(((lo + hi) / 2).sum() / (len(neg) * len(pos)))
