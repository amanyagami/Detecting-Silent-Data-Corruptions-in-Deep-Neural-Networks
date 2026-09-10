"""
Export the real DrDNA detection pipeline for in-browser execution:

  1. ResNet-18 split into two ONNX sub-graphs at the layer1->layer2
     boundary, each also emitting the intermediate activations Dr. DNA
     hooks -- so a browser can run a real forward pass, inject a real
     bit-flip into the real activation tensor between the two graphs, and
     get every activation needed to compute a real tau1/tau2/tau3 score.
  2. The real profile (tau1/tau2/tau3 + DetectionSites.txt) as JSON.
  3. A handful of real CIFAR-10 test images as JSON.
  4. A Python-side reference run (through the *un-split* original model and
     the *split* ONNX graphs) so the browser's JS port can be checked
     against real numbers, not assumed correct.

Run from the repo root (this script imports DRDNA.b10c32.config and
src.models.resnet, so run as `python3 export_browser_demo.py` from
~/drdna-browser).
"""
import copy
import json
import os
import pickle
import random

import onnx
import onnxruntime as ort
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
from src.models.resnet import resnet18

OUT_DIR = "browser_export"
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(0)
torch.manual_seed(0)

device = torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Split model
# ---------------------------------------------------------------------------
class PartA(nn.Module):
    """conv1 -> bn1 -> relu -> maxpool -> layer1 (both blocks).
    Returns the split tensor (input to layer2) plus the 6 Dr. DNA
    activations that live before the split (HOOKED_LAYERS[0:6])."""

    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu, self.maxpool = m.conv1, m.bn1, m.relu, m.maxpool
        self.layer1 = m.layer1

    def forward(self, x):
        x = self.conv1(x)
        a_conv1 = x
        x = self.bn1(x)
        a_bn1 = x
        x = self.relu(x)
        x = self.maxpool(x)

        block = self.layer1[0]
        identity = x
        out = block.conv1(x)
        a_l1c1 = out
        out = block.bn1(out)
        a_l1bn1 = out
        out = block.relu(out)
        out = block.conv2(out)
        a_l1c2 = out
        out = block.bn2(out)
        a_l1bn2 = out
        if block.downsample is not None:
            identity = block.downsample(identity)
        out = out + identity
        out = block.relu(out)

        split = self.layer1[1](out)  # second block of layer1, not hooked
        return split, a_conv1, a_bn1, a_l1c1, a_l1bn1, a_l1c2, a_l1bn2


class PartB(nn.Module):
    """layer2 -> layer3 -> layer4 -> avgpool -> fc, starting from the split
    tensor (possibly corrupted). Returns logits plus the 14 remaining
    Dr. DNA activations (HOOKED_LAYERS[6:20])."""

    def __init__(self, m):
        super().__init__()
        self.layer2, self.layer3, self.layer4 = m.layer2, m.layer3, m.layer4
        self.avgpool, self.fc = m.avgpool, m.fc

    @staticmethod
    def _block(block, x):
        identity = x
        out = block.conv1(x)
        a_c1 = out
        out = block.bn1(out)
        a_bn1 = out
        out = block.relu(out)
        out = block.conv2(out)
        a_c2 = out
        out = block.bn2(out)
        a_bn2 = out
        if block.downsample is not None:
            identity = block.downsample(identity)
        out = out + identity
        out = block.relu(out)
        return out, a_c1, a_bn1, a_c2, a_bn2

    def forward(self, x):
        out, a2c1, a2bn1, a2c2, a2bn2 = self._block(self.layer2[0], x)
        x = self.layer2[1](out)

        out, a3c1, a3bn1, a3c2, a3bn2 = self._block(self.layer3[0], x)
        x = self.layer3[1](out)

        out, a4c1, a4bn1, a4c2, a4bn2 = self._block(self.layer4[0], x)
        x = self.layer4[1](out)

        pooled = self.avgpool(x)
        flat = pooled.reshape(pooled.size(0), -1)
        logits = self.fc(flat)

        return (
            logits,
            a2c1, a2bn1, a2c2, a2bn2,
            a3c1, a3bn1, a3c2, a3bn2,
            a4c1, a4bn1, a4c2, a4bn2,
            pooled, logits,
        )


HOOKED_LAYERS = [
    "conv1", "bn1", "layer1.0.conv1", "layer1.0.bn1", "layer1.0.conv2", "layer1.0.bn2",
    "layer2.0.conv1", "layer2.0.bn1", "layer2.0.conv2", "layer2.0.bn2",
    "layer3.0.conv1", "layer3.0.bn1", "layer3.0.conv2", "layer3.0.bn2",
    "layer4.0.conv1", "layer4.0.bn1", "layer4.0.conv2", "layer4.0.bn2",
    "avgpool", "fc",
]
PART_A_NAMES = HOOKED_LAYERS[:6]
PART_B_NAMES = HOOKED_LAYERS[6:]

print("Loading model...")
model = resnet18(pretrained=True, progress=True).to(device).eval()
part_a = PartA(model).eval()
part_b = PartB(model).eval()

dummy = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    split_out = part_a(dummy)
    split_tensor = split_out[0]
    b_out = part_b(split_tensor)
print("split tensor shape:", tuple(split_tensor.shape))
print("logits shape:", tuple(b_out[0].shape))

part_a_out_names = ["split"] + [f"act_{n}" for n in PART_A_NAMES]
part_b_out_names = ["logits"] + [f"act_{n}" for n in PART_B_NAMES]

torch.onnx.export(
    part_a, dummy, f"{OUT_DIR}/part_a.onnx",
    input_names=["input"], output_names=part_a_out_names,
    opset_version=18, do_constant_folding=True, external_data=False,
)
torch.onnx.export(
    part_b, split_tensor, f"{OUT_DIR}/part_b.onnx",
    input_names=["split"], output_names=part_b_out_names,
    opset_version=18, do_constant_folding=True, external_data=False,
)
onnx.checker.check_model(onnx.load(f"{OUT_DIR}/part_a.onnx"))
onnx.checker.check_model(onnx.load(f"{OUT_DIR}/part_b.onnx"))
print("ONNX export OK, sizes:",
      os.path.getsize(f"{OUT_DIR}/part_a.onnx") / 1e6, "MB /",
      os.path.getsize(f"{OUT_DIR}/part_b.onnx") / 1e6, "MB")


# ---------------------------------------------------------------------------
# 2. Profile -> JSON
# ---------------------------------------------------------------------------
print("Exporting profile...")
PROFILE_DIR = "DRDNA/b10c32"
selected_neurons = []
layer_names = []
with open(os.path.join(PROFILE_DIR, "DetectionSites.txt")) as f:
    for line in f:
        layer_name, location = line.strip().split(",", 1)
        location = eval(location.strip())
        selected_neurons.append((layer_name, location))
        if layer_name not in layer_names:
            layer_names.append(layer_name)
with open(os.path.join(PROFILE_DIR, "tau1.pkl"), "rb") as f:
    tau1 = pickle.load(f)
with open(os.path.join(PROFILE_DIR, "tau2.pkl"), "rb") as f:
    tau2 = pickle.load(f)
with open(os.path.join(PROFILE_DIR, "tau3.pkl"), "rb") as f:
    tau3 = pickle.load(f)


def key_str(layer_name, pos):
    return f"{layer_name}|{','.join(str(int(p)) for p in pos)}"


def scalar(v):
    return int(v.item()) if hasattr(v, "item") else int(v)


# tau1[(layer_name, pos)] is itself a per-*neuron* histogram (bin-range tuple
# -> already-normalized frequency), same shape as tau2's per-*layer*
# histograms (bin-range tuple -> raw count) -- both get the same
# [[ [lo, hi], value ], ...] JSON encoding.
def hist_to_json(hist):
    return [[list(k), v] for k, v in hist.items()]


profile_json = {
    "layer_names": layer_names,
    "selected_neurons": [[ln, list(pos)] for ln, pos in selected_neurons],
    "tau1": {key_str(ln, pos): hist_to_json(tau1[(ln, pos)]) for ln, pos in selected_neurons},
    "tau2": {ln: hist_to_json(tau2[ln]) for ln in layer_names},
    "tau3": {
        ln: {
            "max_value": tau3[ln]["max_value"],
            "max_location": [scalar(x) for x in tau3[ln]["max_location"]],
            "min_value": tau3[ln]["min_value"],
            "min_location": [scalar(x) for x in tau3[ln]["min_location"]],
        }
        for ln in layer_names
    },
}
with open(f"{OUT_DIR}/profile.json", "w") as f:
    json.dump(profile_json, f)
print("profile.json size:", os.path.getsize(f"{OUT_DIR}/profile.json") / 1e6, "MB")


# ---------------------------------------------------------------------------
# 3. Reference scoring (same math as src/detection/pipeline.py) so we can
#    validate the browser's JS port against ground truth below, and so the
#    exported examples carry a real "clean score" a viewer can sanity-check.
# ---------------------------------------------------------------------------
def score_from_activations(activations):
    tau2_running = copy.deepcopy(tau2)
    for ln in tau2_running:
        tau2_running[ln] = reset_histogram(tau2_running[ln])

    tau1_scores = {name: 0.0 for name in layer_names}
    b = c = 0
    for layer_name, pos in selected_neurons:
        if len(pos) > 2:
            b, c, w, h = pos
        else:
            w, h = pos
        value = activations[layer_name][b, c, w, h].item()
        tau1_scores[layer_name] += abnormility_score1(value, tau1[(layer_name, pos)])
        tau2_running[layer_name] = update_histogram(value, tau2_running[layer_name])

    tau2_scores = {ln: abnormality_score2(tau2_running[ln], tau2[ln]) for ln in tau2_running}
    tau3_raw = {ln: t for ln, t in activations.items()}
    tau3_processed = TAU3processing(tau3_raw, 1)
    tau3_scores = {ln: abnormality_score3(tau3_processed[ln], tau3[ln]) for ln in tau3_processed}

    running = 0.0
    for layer in layer_names:
        running += tau1_scores[layer] + tau2_scores[layer] + tau3_scores[layer]
    return running


# ---------------------------------------------------------------------------
# 4. Example images + reference clean scores + a reference faulted score
# ---------------------------------------------------------------------------
print("Building examples + reference scores...")
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
data = torch.load("correct_data.pt", weights_only=False)
inputs, targets = data["inputs"], data["targets"]

N_EXAMPLES = 8
idxs = list(range(0, len(inputs), max(1, len(inputs) // N_EXAMPLES)))[:N_EXAMPLES]

examples = []
ort_a = ort.InferenceSession(f"{OUT_DIR}/part_a.onnx")
ort_b = ort.InferenceSession(f"{OUT_DIR}/part_b.onnx")

for idx in idxs:
    img = inputs[idx : idx + 1]
    label = int(targets[idx].item())

    with torch.no_grad():
        torch_out_a = part_a(img)
        split = torch_out_a[0]
        torch_out_b = part_b(split)
    activations = dict(zip(PART_A_NAMES, torch_out_a[1:]))
    activations.update(dict(zip(PART_B_NAMES, torch_out_b[1:])))
    clean_score = score_from_activations(activations)
    logits = torch_out_b[0]
    pred = int(logits.argmax(1).item())

    # cross-check: ONNX split produces the same activations as PyTorch
    ort_out_a = ort_a.run(None, {"input": img.numpy()})
    import numpy as np

    max_diff_a = max(
        float(np.abs(ort_out_a[i] - torch_out_a[i].detach().numpy()).max()) for i in range(len(torch_out_a))
    )
    ort_out_b = ort_b.run(None, {"split": ort_out_a[0]})
    max_diff_b = max(
        float(np.abs(ort_out_b[i] - torch_out_b[i].detach().numpy()).max()) for i in range(len(torch_out_b))
    )

    examples.append(
        {
            "index": idx,
            "label": CIFAR10_CLASSES[label],
            "pixels": img.numpy().reshape(-1).tolist(),  # 3*32*32 floats, CHW
            "clean_score": clean_score,
            "clean_pred": CIFAR10_CLASSES[pred],
            "onnx_max_diff_part_a": max_diff_a,
            "onnx_max_diff_part_b": max_diff_b,
        }
    )
    print(
        f"  idx={idx} label={CIFAR10_CLASSES[label]} pred={CIFAR10_CLASSES[pred]} "
        f"clean_score={clean_score:.2f} onnx_diff=({max_diff_a:.2e},{max_diff_b:.2e})"
    )

with open(f"{OUT_DIR}/examples.json", "w") as f:
    json.dump(examples, f)
print("examples.json size:", os.path.getsize(f"{OUT_DIR}/examples.json") / 1e6, "MB")

print("\nDone. Files in", OUT_DIR, ":", os.listdir(OUT_DIR))
