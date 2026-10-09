import math

import numpy as np
import pytest
import torch
from torch import nn

from drdna import DrDNA, NeuronFault, flip_bit
from drdna.evaluate import bit_class
from drdna.stats import auroc, wilson


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))


def make(seed: int = 0):
    torch.manual_seed(seed)
    model = Tiny().eval()
    det = DrDNA(["conv1", "bn1", "conv2"], cohort=16, bins=10, seed=seed)
    data = [torch.randn(64, 3, 8, 8) for _ in range(4)]
    det.fit(model, iter(data))
    return model, det


def test_flip_bit_known_values() -> None:
    one = torch.tensor([1.0])
    assert flip_bit(one, 31).item() == -1.0
    assert flip_bit(one, 30).item() > 1e30  # exponent MSB
    assert flip_bit(one, 0).item() == pytest.approx(1.0, abs=2e-7)
    assert flip_bit(one, 0).item() != 1.0


@pytest.mark.parametrize("bit", [0, 5, 22, 23, 30, 31])
def test_flip_is_involution(bit: int) -> None:
    x = torch.randn(50)
    assert torch.equal(flip_bit(flip_bit(x, bit), bit), x)


def test_flip_validation() -> None:
    with pytest.raises(ValueError):
        flip_bit(torch.ones(1), 32)
    with pytest.raises(TypeError):
        flip_bit(torch.ones(1, dtype=torch.float64), 3)


def test_fault_changes_only_target_and_hook_is_removed() -> None:
    model, _ = make()
    x = torch.randn(2, 3, 8, 8)
    base = {}
    h = model.conv1.register_forward_hook(lambda m, i, o: base.update(o=o.detach().clone()))
    model(x)
    h.remove()
    seen = {}
    with NeuronFault(model, "conv1", 2, 3, 4, 31):
        # registered after the fault, so it observes the corrupted output (as a detector does)
        h2 = model.conv1.register_forward_hook(lambda m, i, o: seen.update(o=o.detach().clone()))
        model(x)
        h2.remove()
    diff = (seen["o"] != base["o"]).nonzero()
    assert {tuple(d.tolist()[1:]) for d in diff} == {(2, 3, 4)}
    assert len(model.conv1._forward_hooks) == 0


def test_clean_scores_are_low_and_spike_is_flagged() -> None:
    model, det = make()
    clean = torch.randn(256, 3, 8, 8)
    _, s_clean = det.score(model, clean)
    base = s_clean.cumulative()[:, -1]
    # corrupt the whole layer output, so the cohort definitely sees it
    h = model.conv1.register_forward_hook(lambda m, i, o: o * 1e9)
    _, s_bad = det.score(model, clean)
    h.remove()
    bad = s_bad.cumulative()[:, -1]
    assert bad.min() > base.max()


def test_nonfinite_activations_score_as_maximally_abnormal() -> None:
    model, det = make()
    x = torch.randn(8, 3, 8, 8)
    _, clean = det.score(model, x)
    h = model.conv1.register_forward_hook(lambda m, i, o: o * float("nan"))
    _, nan = det.score(model, x)
    h.remove()
    assert torch.isfinite(nan.tau1).all() and torch.isfinite(nan.tau2).all()
    assert (nan.tau1[:, 0] == 16).all()  # every cohort site out of range
    assert nan.cumulative()[:, -1].min() > clean.cumulative()[:, -1].max()


def test_scores_shapes_and_ranges() -> None:
    model, det = make()
    _, s = det.score(model, torch.randn(10, 3, 8, 8))
    for t in (s.tau1, s.tau2, s.tau3):
        assert t.shape == (10, 3)
    assert (s.tau1 >= 0).all() and (s.tau1 <= 16).all()
    assert (s.tau2 >= 0).all() and (s.tau2 <= 1).all()
    assert set(s.tau3.unique().tolist()) <= {0.0, 0.5, 1.0}
    cum = s.cumulative()
    assert (cum[:, 1:] >= cum[:, :-1]).all()  # accumulates with depth


def test_ablation_mask_selects_components() -> None:
    model, det = make()
    _, s = det.score(model, torch.randn(4, 3, 8, 8))
    assert torch.allclose(s.cumulative(mask="1"), torch.cumsum(s.tau1, 1))
    assert torch.allclose(s.cumulative((1.0, 1.0, 1.0), "13"), torch.cumsum(s.tau1 + s.tau3, 1))


def test_deterministic_given_seed() -> None:
    m1, d1 = make(3)
    m2, d2 = make(3)
    x = torch.randn(5, 3, 8, 8)
    _, a = d1.score(m1, x)
    _, b = d2.score(m2, x)
    assert torch.equal(a.cumulative(), b.cumulative())


def test_score_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        DrDNA(["conv1"]).score(Tiny(), torch.randn(1, 3, 8, 8))


def test_stats_helpers() -> None:
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    lo, hi = wilson(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and math.isclose((lo + hi) / 2, 0.5, abs_tol=0.01)
    assert (
        bit_class(31) == "sign" and bit_class(30) == "exponent" and bit_class(3) == "mantissa-low"
    )
