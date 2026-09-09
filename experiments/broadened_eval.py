"""
Broadened evaluation beyond the single ResNet-18 / CIFAR-10 / single-bit-flip
slice this reproduction was originally validated on. Two axes, both reusing
the already-validated model/profile rather than training anything new:

  1. A second *dataset*: the same CIFAR-10-trained ResNet-18 and CIFAR-10
     activation profile, evaluated on SVHN inputs (correct_data_SVHN.pt) --
     a cross-domain stress test of whether the profiled baseline still
     applies once the input distribution shifts.
  2. A second *error model*: multi-bit-flip (two bits on one neuron) instead
     of single-bit-flip, on the original CIFAR-10 setup.

Writes real per-sample scores to CSV (same one-column format as
testwitSDCoutput.csv / testwithoutSDC.csv) and prints summary stats +
TPR/FPR at the repo's calibrated threshold (499) for each combination.

Usage: python -m experiments.broadened_eval [--n-clean N] [--n-fault N] [--out-dir DIR]
"""
import argparse
import csv
import statistics

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.detection import Profile, evaluate, keep_all, keep_correct, keep_misclassified
from src.faults import random_neuron_multi_bit_inj_Aman, random_neuron_single_bit_inj_Aman, single_bit_flip_func
from src.models.resnet import resnet18

FI_LAYER, FI_C, FI_H, FI_W = 5, 6, 7, 2
RANGES = [9999] * 6 + [999999999] + [9999] * 13  # matches DRDNA/b10c32/TestwithSDC.py
THRESHOLD = 499  # this repo's calibrated single-bit-flip/CIFAR-10 margin


def load_model(device):
    return resnet18(pretrained=True, progress=True).to(device)


def load_loader(path, n):
    data = torch.load(path, weights_only=False)
    inputs, targets = data["inputs"][:n], data["targets"][:n]
    return DataLoader(TensorDataset(inputs, targets), batch_size=1, shuffle=False, num_workers=0)


def write_csv(path, scores):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for s in scores:
            writer.writerow([s])


def summarize(name, scores):
    if not scores:
        print(f"  {name}: 0 samples")
        return
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    print(f"  {name}: n={len(scores)} mean={statistics.mean(scores):.2f} std={std:.2f} "
          f"min={min(scores):.2f} max={max(scores):.2f}")


def detection_stats(clean, faulted, threshold=THRESHOLD):
    if not clean or not faulted:
        print(f"  threshold={threshold}: insufficient samples to compute TPR/FPR")
        return None, None
    tp = sum(1 for s in faulted if s > threshold)
    fp = sum(1 for s in clean if s > threshold)
    tpr, fpr = tp / len(faulted) * 100, fp / len(clean) * 100
    print(f"  threshold={threshold}: TPR={tpr:.1f}% ({tp}/{len(faulted)})  FPR={fpr:.1f}% ({fp}/{len(clean)})")
    return tpr, fpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clean", type=int, default=200)
    ap.add_argument("--n-fault", type=int, default=200)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile = Profile.load("DRDNA/b10c32")

    print("=== Axis 1: SVHN (cross-domain), same CIFAR-10-trained model + profile ===")
    svhn_clean_model = load_model(device)
    svhn_scores = evaluate(
        svhn_clean_model, load_loader("correct_data_SVHN.pt", args.n_clean), profile, device,
        keep=keep_all, max_samples=args.n_clean,
    )
    summarize("SVHN, no fault", svhn_scores)
    write_csv(f"{args.out_dir}/svhn_no_fault_scores.csv", svhn_scores)

    svhn_fault_model = load_model(device)
    pfi_svhn = single_bit_flip_func(
        svhn_fault_model, batch_size=1, input_shape=[3, 32, 32],
        use_cuda=torch.cuda.is_available(), bits=8, random_batch=False, bit_pos=1,
    )
    faulted_svhn_model = random_neuron_single_bit_inj_Aman(pfi_svhn, RANGES, FI_LAYER, FI_C, FI_H, FI_W, bit_pos=1)
    svhn_fault_scores = evaluate(
        faulted_svhn_model, load_loader("correct_data_SVHN.pt", args.n_fault), profile, device,
        keep=keep_all, max_samples=args.n_fault, on_fault_step=pfi_svhn.reset_current_layer,
    )
    summarize("SVHN, single-bit-flip", svhn_fault_scores)
    write_csv(f"{args.out_dir}/svhn_single_bit_flip_scores.csv", svhn_fault_scores)
    detection_stats(svhn_scores, svhn_fault_scores)

    print("\n=== Axis 2: CIFAR-10, multi-bit-flip (bits 1 & 9) vs. clean ===")
    cifar_clean_model = load_model(device)
    clean_scores = evaluate(
        cifar_clean_model, load_loader("correct_data.pt", args.n_clean), profile, device,
        keep=keep_correct, max_samples=args.n_clean,
    )
    summarize("CIFAR-10, no fault", clean_scores)
    write_csv(f"{args.out_dir}/cifar10_clean_scores.csv", clean_scores)

    multi_fault_model = load_model(device)
    pfi_multi = single_bit_flip_func(
        multi_fault_model, batch_size=1, input_shape=[3, 32, 32],
        use_cuda=torch.cuda.is_available(), bits=8, random_batch=False, bit_pos=1,
    )
    faulted_multi_model = random_neuron_multi_bit_inj_Aman(
        pfi_multi, RANGES, FI_LAYER, FI_C, FI_H, FI_W, bit_positions=[1, 9]
    )
    multi_scores = evaluate(
        faulted_multi_model, load_loader("correct_data.pt", args.n_fault), profile, device,
        keep=keep_misclassified, max_samples=args.n_fault, on_fault_step=pfi_multi.reset_current_layer,
    )
    summarize("CIFAR-10, multi-bit-flip", multi_scores)
    write_csv(f"{args.out_dir}/cifar10_multi_bit_flip_scores.csv", multi_scores)
    detection_stats(clean_scores, multi_scores)


if __name__ == "__main__":
    main()
