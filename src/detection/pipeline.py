"""
Shared Dr. DNA detection pipeline.

`DRDNA/b10c32/TestwithSDC.py` and `DRDNA/b10c32/testwithoutSDC.py` each
implemented their own ~90%-duplicated copy of: hook registration, per-sample
tau1/tau2/tau3 scoring, and a CSV-writing evaluation loop. This module is
that logic in one place, as importable functions, so it can be unit-tested
and reused (e.g. for a new dataset/model pair) instead of being copy-pasted
into a third script.

NOTE on fidelity: `score_sample()` intentionally reproduces the *exact* same
per-neuron indexing as the original scripts, including a subtlety in how a
detection site with only 2 stored coordinates (e.g. the `fc` layer, whose
activation tensor is 2-D) is indexed -- the original code reuses whatever
`b, c` were last bound to by the *previous* 4-D neuron in the loop, rather
than deriving them fresh. With batch_size=1 (as used everywhere in this
repo) this happens to be harmless for `b`, and empirically it hasn't changed
the validated clean-vs-SDC score separation, but it's worth flagging rather
than silently "fixing" -- changing it would be a numerical behavior change
that needs its own dedicated validation, not a drive-by refactor.
"""
from __future__ import annotations

import copy
import os
import pickle
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from DRDNA.b10c32.config import (
    TAU3processing,
    abnormality_score2,
    abnormality_score3,
    abnormility_score1,
    reset_histogram,
    update_histogram,
)

# The 20 layers Dr. DNA hooks in this ResNet-18 reproduction (conv1/bn1,
# the first conv+bn pair of each of the 4 residual stages, avgpool, fc).
HOOKED_LAYERS = [
    "conv1",
    "bn1",
    "layer1.0.conv1", "layer1.0.bn1", "layer1.0.conv2", "layer1.0.bn2",
    "layer2.0.conv1", "layer2.0.bn1", "layer2.0.conv2", "layer2.0.bn2",
    "layer3.0.conv1", "layer3.0.bn1", "layer3.0.conv2", "layer3.0.bn2",
    "layer4.0.conv1", "layer4.0.bn1", "layer4.0.conv2", "layer4.0.bn2",
    "avgpool",
    "fc",
]


@dataclass
class Profile:
    """A Dr. DNA activation profile -- the clean-run baseline (tau1/tau2/tau3
    plus the cohort of detection-site neurons) that inference-time scores
    are compared against. Produced offline by `offlineProfiling.py`."""

    tau1: dict
    tau2: dict
    tau3: dict
    selected_neurons: List[Tuple[str, tuple]]
    layer_names: List[str]

    @classmethod
    def load(cls, profile_dir: str) -> "Profile":
        selected_neurons: List[Tuple[str, tuple]] = []
        layer_names: List[str] = []
        with open(os.path.join(profile_dir, "DetectionSites.txt")) as f:
            for line in f:
                layer_name, location = line.strip().split(",", 1)
                location = eval(location.strip())  # noqa: S307 -- trusted, repo-authored file
                selected_neurons.append((layer_name, location))
                if layer_name not in layer_names:
                    layer_names.append(layer_name)
        with open(os.path.join(profile_dir, "tau1.pkl"), "rb") as f:
            tau1 = pickle.load(f)
        with open(os.path.join(profile_dir, "tau2.pkl"), "rb") as f:
            tau2 = pickle.load(f)
        with open(os.path.join(profile_dir, "tau3.pkl"), "rb") as f:
            tau3 = pickle.load(f)
        return cls(tau1=tau1, tau2=tau2, tau3=tau3, selected_neurons=selected_neurons, layer_names=layer_names)


def register_hooks(model: nn.Module) -> dict:
    """Attach a forward hook at every `HOOKED_LAYERS` module and return the
    dict those hooks write captured activations into (keyed by layer name)."""
    activations: dict = {}

    def _make_hook(name: str):
        def hook(_module, _input, output):
            activations[name] = output.detach()

        return hook

    model.conv1.register_forward_hook(_make_hook("conv1"))
    model.bn1.register_forward_hook(_make_hook("bn1"))
    for stage in (1, 2, 3, 4):
        block = getattr(model, f"layer{stage}")[0]
        block.conv1.register_forward_hook(_make_hook(f"layer{stage}.0.conv1"))
        block.bn1.register_forward_hook(_make_hook(f"layer{stage}.0.bn1"))
        block.conv2.register_forward_hook(_make_hook(f"layer{stage}.0.conv2"))
        block.bn2.register_forward_hook(_make_hook(f"layer{stage}.0.bn2"))
    model.avgpool.register_forward_hook(_make_hook("avgpool"))
    model.fc.register_forward_hook(_make_hook("fc"))
    return activations


def score_sample(activations: dict, profile: Profile) -> float:
    """Compute one forward pass's cumulative Dr. DNA abnormality score:
    Score[l] = tau1[l] + tau2[l] + tau3[l] + Score[l-1], summed over every
    hooked layer in `profile.layer_names` order (lambda1=lambda2=lambda3=1,
    matching `DRDNA/b10c32/config.py`)."""
    tau2_running = copy.deepcopy(profile.tau2)
    for layer_name in tau2_running:
        tau2_running[layer_name] = reset_histogram(tau2_running[layer_name])

    tau1_scores = {name: 0.0 for name in profile.layer_names}
    b = c = 0  # noqa: E741 -- see the module docstring: intentionally preserved quirk
    for layer_name, pos in profile.selected_neurons:
        if len(pos) > 2:
            b, c, w, h = pos
        else:
            w, h = pos
        value = activations[layer_name][b, c, w, h].item()
        tau1_scores[layer_name] += abnormility_score1(value, profile.tau1[(layer_name, pos)])
        tau2_running[layer_name] = update_histogram(value, tau2_running[layer_name])

    tau2_scores = {name: abnormality_score2(tau2_running[name], profile.tau2[name]) for name in tau2_running}

    tau3_raw = {name: tensor for name, tensor in activations.items()}
    tau3_processed = TAU3processing(tau3_raw, 1)
    tau3_scores = {name: abnormality_score3(tau3_processed[name], profile.tau3[name]) for name in tau3_processed}

    running = 0.0
    for layer in profile.layer_names:
        running = running + tau1_scores[layer] + tau2_scores[layer] + tau3_scores[layer]
    return running


def evaluate(
    model: nn.Module,
    loader,
    profile: Profile,
    device: torch.device,
    keep: Callable[[torch.Tensor, torch.Tensor], bool],
    max_samples: Optional[int] = None,
    on_fault_step: Optional[Callable[[], None]] = None,
) -> List[float]:
    """Run `model` over `loader`, scoring every sample against `profile`, and
    return the Dr. DNA score for each sample where `keep(predicted, targets)`
    is true.

    This is the loop shared by the clean-run and fault-injected-run scripts:
      - clean run:    keep = correctly-classified samples (the profiling
                       baseline should score low on samples the model still
                       gets right).
      - faulted run:  keep = misclassified samples (a "successful" SDC --
                       the fault actually changed the model's answer, which
                       is the case Dr. DNA needs to catch).

    `on_fault_step` is called once per sample before the forward pass, e.g.
    to advance a PyTorchFI injector's internal layer counter.
    """
    model.eval()
    activations = register_hooks(model)
    scores: List[float] = []
    criterion = nn.CrossEntropyLoss()
    test_loss, total = 0.0, 0
    n_seen = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if on_fault_step is not None:
                on_fault_step()

            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            if keep(predicted, targets):
                scores.append(score_sample(activations, profile))

            n_seen += 1
            if max_samples is not None and n_seen >= max_samples:
                break

    return scores


def keep_correct(predicted: torch.Tensor, targets: torch.Tensor) -> bool:
    """Selects samples the model still classifies correctly -- used for the
    clean-run baseline."""
    return bool(predicted.eq(targets).sum().item() == targets.size(0))


def keep_misclassified(predicted: torch.Tensor, targets: torch.Tensor) -> bool:
    """Selects samples where the model's prediction flipped -- a
    "successful" SDC, which is the event Dr. DNA is being asked to detect."""
    return bool(predicted.eq(targets).sum().item() == 0)


def keep_all(_predicted: torch.Tensor, _targets: torch.Tensor) -> bool:
    """Keeps every sample regardless of prediction correctness. Needed when
    "correct" isn't a meaningful filter -- e.g. scoring a CIFAR-10-profiled
    model on SVHN inputs, where its 10-way output has no real relationship
    to SVHN's digit labels, so gating on classification accuracy would be
    gating on noise."""
    return True
