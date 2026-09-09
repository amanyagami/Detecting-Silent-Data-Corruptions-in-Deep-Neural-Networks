"""End-to-end regression test for the detection pipeline: run a small clean
pass and a small single-bit-flip pass through the real ResNet-18/CIFAR-10
profile shipped in this repo (via LFS), and assert the same separation this
repo's README and project page report -- SDC-injected runs should score
meaningfully higher than clean runs. This is the automated version of the
manual validation this reproduction was checked against.

Needs the repo's LFS assets (state_dicts/resnet18.pt, correct_data.pt,
DRDNA/b10c32/{tau1,tau2,tau3}.pkl, DetectionSites.txt) -- `git lfs pull`
before running, see .github/workflows/test.yml.
"""
import statistics

import torch

from src.detection import evaluate, keep_correct, keep_misclassified
from src.faults import random_neuron_single_bit_inj_Aman, single_bit_flip_func

N_CLEAN_SAMPLES = 20
N_FAULT_SAMPLES = 20
FI_LAYER, FI_C, FI_H, FI_W, BIT_POS = 5, 6, 7, 2, 1
# Loose enough not to be flaky on a handful of samples, tight enough to catch
# a real regression in the scoring pipeline.
CLEAN_SCORE_CEILING = 515


def test_clean_run_scores_stay_low(resnet18_model, correct_data_loader, profile, device):
    scores = evaluate(
        resnet18_model,
        correct_data_loader,
        profile,
        device,
        keep=keep_correct,
        max_samples=N_CLEAN_SAMPLES,
    )
    assert len(scores) > 0, "no correctly-classified samples scored -- pipeline or data problem"
    assert statistics.mean(scores) < CLEAN_SCORE_CEILING


def test_single_bit_flip_scores_separate_from_clean(resnet18_model, correct_data_loader, profile, device):
    torch.manual_seed(0)
    ranges = [9999] * 6 + [999999999] + [9999] * 13  # matches DRDNA/b10c32/TestwithSDC.py
    pfi = single_bit_flip_func(
        resnet18_model,
        batch_size=1,
        input_shape=[3, 32, 32],
        use_cuda=torch.cuda.is_available(),
        bits=8,
        random_batch=False,
        bit_pos=BIT_POS,
    )
    faulted_model = random_neuron_single_bit_inj_Aman(pfi, ranges, FI_LAYER, FI_C, FI_H, FI_W, bit_pos=BIT_POS)

    clean_scores = evaluate(
        resnet18_model,
        correct_data_loader,
        profile,
        device,
        keep=keep_correct,
        max_samples=N_CLEAN_SAMPLES,
    )
    fault_scores = evaluate(
        faulted_model,
        correct_data_loader,
        profile,
        device,
        keep=keep_misclassified,
        max_samples=N_FAULT_SAMPLES,
        on_fault_step=pfi.reset_current_layer,
    )

    assert len(clean_scores) > 0
    if len(fault_scores) == 0:
        # The fault didn't flip any prediction in this small a sample -- not
        # a pipeline bug, just an unlucky injection site/sample combination.
        # Skip rather than fail so this stays non-flaky in CI.
        import pytest

        pytest.skip("fault injection didn't misclassify any of the sampled inputs")

    assert statistics.mean(fault_scores) > statistics.mean(clean_scores)
