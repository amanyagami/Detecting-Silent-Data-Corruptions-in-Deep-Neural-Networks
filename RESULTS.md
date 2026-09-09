# Results: broadened evaluation

The original reproduction covers one configuration: ResNet-18, CIFAR-10,
single-bit-flip. This extends it along two axes that reuse the same
already-validated model and activation profile — no new training — rather
than starting a full re-run of the paper's own 10-model × 5-dataset ×
4-error-model matrix, which this reproduction does not attempt.

Numbers below are computed by `experiments/broadened_eval.py`, from the raw
per-sample CSVs it writes (also checked in under `results/`), the same way
the original single-bit-flip/CIFAR-10 numbers in the README were computed
from `testwitSDCoutput.csv` / `testwithoutSDC.csv`.

## Axis 1 — a second dataset (SVHN), same model + profile

Same CIFAR-10-trained ResNet-18 and CIFAR-10 activation profile, run on SVHN
inputs (`correct_data_SVHN.pt`) instead. This isn't asking "does the model
classify SVHN well" (it wasn't trained to, and `keep_all` is used instead of
filtering by classification correctness) — it's asking whether a profile
built on one input distribution still gives a usable detection signal on a
different one.

| | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| No fault | 300 | 491.6 | 6.1 | 474.9 | 510.6 |
| Single-bit-flip | 300 | 500.9 | 16.0 | 480.5 | 538.8 |

At the CIFAR-10-calibrated margin (499): **TPR 35.7% (107/300), FPR 11.7% (35/300)**.

**This is a real negative result, and it's the most useful finding in this
file.** On CIFAR-10 the same threshold gets ~99.5% TPR at ~5.6% FPR (see
README.md); on SVHN it barely beats a coin flip on true positives and gives
2x the false-positive rate. The clean-run mean itself shifts from ~487.7
(CIFAR-10) to ~491.6 (SVHN) and its spread roughly doubles (std 6.1 vs
~6.2 is close, but the *faulted* distribution's std balloons from ~9.9 on
CIFAR-10 to 16.0 here) — the profiled "normal" baseline is itself a worse
fit for out-of-distribution inputs, which eats into the margin between
clean and faulted scores. Concretely: **an activation-distribution profile
trained on one input domain does not transfer to another one**, at least
not with a fixed threshold. A deployed version of this detector would need
either a per-domain profile or a recalibrated threshold per input
distribution -- it isn't a universal fault signature.

## Axis 2 — a second error model (multi-bit-flip), same CIFAR-10 setup

Same model, profile, and dataset as the original reproduction; the fault
model flips two bits (positions 1 and 9) on one neuron's activation instead
of one bit.

| | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| No fault | 300 | 487.7 | 6.2 | 474.2 | 508.8 |
| Multi-bit-flip (2 bits) | 103\* | 532.3 | 8.2 | 490.7 | 542.6 |

\* 300 injections attempted; 103 actually flipped the model's prediction
(the event Dr. DNA is asked to catch -- see `keep_misclassified` in
`src/detection/pipeline.py`). A 2-bit flip on this particular neuron/layer
is evidently not much more likely to change the prediction than a 1-bit
flip is; severity of the corruption and probability of it mattering to the
output aren't the same thing.

At the CIFAR-10-calibrated margin (499): **TPR 97.1% (100/103), FPR 5.3%
(16/300)**. This is close to the single-bit-flip result on the same
model/dataset (~99.5% TPR / ~5.6% FPR) -- the *existing* single-bit-flip
threshold, never recalibrated for this error model, still works almost as
well on multi-bit-flip corruptions. Unlike the SVHN case, the error-model
axis generalizes; the dataset/domain axis does not.

## Comparison to the original paper

| | Dr. DNA (paper, full matrix) | This repo (original) | This repo (broadened) |
|---|---|---|---|
| | Dr. DNA (paper) | This repo, original slice | This repo, + SVHN | This repo, + multi-bit-flip |
|---|---|---|---|---|
| TPR | 100% most cases, 95% avg, >90% all cases | ~99.5% (n=3,596) | 35.7% (n=300) | 97.1% (n=103) |
| FPR | not reported in the abstract | ~5.6% (n=3,596) | 11.7% (n=300) | 5.3% (n=300) |
| Scope | 3 tasks, 5 datasets, 10 models, 4 error models | 1 model, 1 dataset, 1 error model | same model/profile, 2nd dataset | same model/dataset, 2nd error model |
| Mitigation | Yes (<1% memory, <2.5% latency overhead) | Not implemented | Not implemented | Not implemented |

Headline takeaway: **the error model generalized, the dataset didn't.** This
is still a small slice of the paper's full evaluation matrix, not a full
reproduction — see the scope note in `README.md`. Raw per-sample scores for
every row above are in `results/*.csv`; regenerate with
`python -m experiments.broadened_eval`.
