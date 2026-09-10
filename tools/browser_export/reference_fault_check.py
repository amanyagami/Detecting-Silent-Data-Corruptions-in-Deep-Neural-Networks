"""
Generate ground-truth *faulted* scores for a few (example, fault-site)
combinations, using the real PyTorch model (not the ONNX export) and the
exact same bit-flip math as src/utils/customFI_methods.py. These become the
"did the browser's JS get the real answer" checks for the live demo.

Run from ~/drdna-browser after export_browser_demo.py.
"""
import copy
import json
import pickle
import struct

import torch

from DRDNA.b10c32.config import (
    TAU3processing,
    abnormality_score2,
    abnormality_score3,
    abnormility_score1,
    reset_histogram,
    update_histogram,
)
from src.models.resnet import resnet18

device = torch.device("cpu")

# Re-declare the exact same split as export_browser_demo.py (kept separate,
# tiny script, not worth importing across files for this one-off check).
import importlib.util

spec = importlib.util.spec_from_file_location("export_browser_demo", "export_browser_demo.py")
# don't re-run the whole export (slow + re-writes files); just grab the classes
import types

mod = types.ModuleType("export_browser_demo_defs")
src = open("export_browser_demo.py").read()
# execute only up to (not including) the "Loading model..." print, i.e. just class defs + HOOKED_LAYERS
cut = src.index('print("Loading model...")')
exec(src[:cut], mod.__dict__)

model = resnet18(pretrained=True, progress=True).to(device).eval()
part_a = mod.PartA(model).eval()
part_b = mod.PartB(model).eval()
PART_A_NAMES, PART_B_NAMES, layer_names_const = mod.PART_A_NAMES, mod.PART_B_NAMES, mod.HOOKED_LAYERS

PROFILE_DIR = "DRDNA/b10c32"
selected_neurons, layer_names = [], []
with open(f"{PROFILE_DIR}/DetectionSites.txt") as f:
    for line in f:
        ln, loc = line.strip().split(",", 1)
        loc = eval(loc.strip())
        selected_neurons.append((ln, loc))
        if ln not in layer_names:
            layer_names.append(ln)
tau1 = pickle.load(open(f"{PROFILE_DIR}/tau1.pkl", "rb"))
tau2 = pickle.load(open(f"{PROFILE_DIR}/tau2.pkl", "rb"))
tau3 = pickle.load(open(f"{PROFILE_DIR}/tau3.pkl", "rb"))


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


def flip_bit_ieee754(value: float, bit_pos: int) -> float:
    """Bit-identical to customFI_methods.py's _flip_bit_signed_Aman: pack as
    big-endian float32, flip one bit (0 = sign, MSB-first), unpack."""
    packed = struct.pack("!f", value)
    as_int = int.from_bytes(packed, "big")
    bits = list(bin(as_int)[2:].zfill(32))
    bits[bit_pos] = "1" if bits[bit_pos] == "0" else "0"
    new_int = int("".join(bits), 2)
    return struct.unpack("!f", new_int.to_bytes(4, "big"))[0]


data = torch.load("correct_data.pt", weights_only=False)
inputs, targets = data["inputs"], data["targets"]

cases = [
    {"idx": 0, "c": 6, "h": 7, "w": 2, "bit_pos": 1},
    {"idx": 1163, "c": 30, "h": 3, "w": 9, "bit_pos": 0},
    {"idx": 2326, "c": 50, "h": 10, "w": 10, "bit_pos": 8},
]

results = []
for case in cases:
    img = inputs[case["idx"] : case["idx"] + 1]
    with torch.no_grad():
        out_a = part_a(img)
    split = out_a[0].clone()
    pre_activations = dict(zip(PART_A_NAMES, out_a[1:]))

    orig_val = split[0, case["c"], case["h"], case["w"]].item()
    new_val = flip_bit_ieee754(orig_val, case["bit_pos"])
    split[0, case["c"], case["h"], case["w"]] = new_val

    with torch.no_grad():
        out_b = part_b(split)
    post_activations = dict(zip(PART_B_NAMES, out_b[1:]))
    activations = {**pre_activations, **post_activations}
    faulted_score = score_from_activations(activations)
    pred = int(out_b[0].argmax(1).item())

    results.append(
        {
            **case,
            "orig_value": orig_val,
            "corrupted_value": new_val,
            "faulted_score": faulted_score,
            "faulted_pred_class_idx": pred,
        }
    )
    print(case, "-> orig", orig_val, "corrupted", new_val, "score", faulted_score, "pred_idx", pred)

with open("browser_export/reference_faults.json", "w") as f:
    json.dump(results, f, indent=2)
print("wrote browser_export/reference_faults.json")
