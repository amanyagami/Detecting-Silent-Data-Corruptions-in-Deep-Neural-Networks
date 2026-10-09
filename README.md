# Dr. DNA (unofficial reproduction) - detection path

Unofficial reproduction of the detection path of **"Dr. DNA: Combating Silent Data Corruptions in
Deep Learning using Distribution of Neuron Activations"** (Ma et al., ASPLOS '24) on ResNet-18 /
CIFAR-10 with a single-bit-flip error model. Not affiliated with the authors; please cite the paper.

## Quick start

```bash
git lfs pull                       # fetches state_dicts/resnet18.pt (it is stored with Git LFS)
uv sync --extra plots
uv run pytest tests/test_drdna.py
uv run drdna-eval --cifar-root <dir containing cifar-10-batches-py> --faults 300
# only exponent-bit flips (the ones that cause SDCs), 400 faults x 8 images:
uv run drdna-eval --cifar-root <dir> --faults 400 --images-per-fault 8 --bit-min 23 --bit-max 30 --tag _exponent
```

```python
from drdna import DrDNA, NeuronFault
from drdna.detector import default_layers
from drdna.model import cifar_resnet18

model = cifar_resnet18("state_dicts/resnet18.pt")
det = DrDNA(default_layers(model)).fit(model, profile_batches)       # fault-free profiling
with NeuronFault(model, "layer2.0.conv1", channel=5, row=3, col=1, bit=30):
    logits, scores = det.score(model, x)                              # fault is observed by the detector
flag = scores.cumulative()[:, -1] > threshold                         # threshold from clean data
```

## Results (this protocol)

Profile on 1,000 training images; choose the threshold as the 99th percentile of the final score on
2,000 separate clean test images; evaluate on other test images the model classifies correctly.
Faults are random layer / neuron / bit, and each trial is labelled by what it did to the output:
*masked* (top-1 unchanged), *SDC* (top-1 changed) or *non-finite*. Results are in `results/`.

| campaign | trials | masked | SDC | detector (FPR budget 1%) | measured FPR (clean) | SDC detected (95% CI) |
|---|---|---|---|---|---|---|
| random bit 0-31, 300 faults x 16 images | 4,800 | 4,761 | 39 | tau2 | 0.8% | 97.4% [86.8, 99.5] |
| | | | | tau1/k + tau2 | 0.4% | 97.4% [86.8, 99.5] |
| | | | | tau1 + tau2 + tau3 (lambda = 1) | 0.4% | 89.7% [76.4, 95.9] |
| exponent bits 23-30, 400 faults x 8 images | 3,200 | 3,060 | 139 | tau2 | 0.9% | 90.6% [84.7, 94.5] |
| | | | | tau1/k + tau2 | 0.5% | 87.8% [81.3, 92.2] |
| | | | | tau1 + tau2 + tau3 (lambda = 1) | 0.3% | 84.9% [78.0, 89.9] |

What this says:

* **Most flips do nothing.** 99% of random single-bit flips were masked; **every SDC came from an
  exponent bit** (sign and mantissa flips never changed a prediction in these trials). Exponent-only
  sampling is the efficient way to measure detection.
* **The original "100% detected" is not reproduced** under this stricter protocol: about 85-97%
  of SDCs are detected at roughly 0.3-0.9% false positives, with intervals that exclude 100%. The
  original measured one fixed neuron (layer index 5, c=6, h=7, w=2) and reported no false-positive
  rate, so the two numbers are not the same experiment.
* **tau3 carries no signal as the README originally defined it** (AUROC 0.42 and 0.62; it fires
  for essentially every input, clean or faulty, because the argmax location of a single input rarely
  equals the argmax of the profile mean). Adding it with lambda = 1 only adds noise.
* **tau1 is a sum over 32 sites (0-32), tau2 lies in 0-1**, so lambda = 1 lets tau1 dominate and the
  combined score is worse than tau2 alone. Dividing tau1 by the cohort size recovers the gain.
* **Late faults are missed.** SDCs injected in `layer4.0.conv2`, `layer4.0.bn2` and `avgpool` were
  never detected (0 of 18 in the exponent campaign): detection works through propagation into the
  sampled sites, and a corrupted neuron near the output has little left to disturb.
* Calibration transfers to unseen clean data conservatively: at a 5% budget the measured false-positive rate was 3.1-3.4%, and at 1% it was 0.3-0.9%.

## What changed relative to the original scripts

| Original | `drdna` package |
|---|---|
| Threshold taken from the faulty runs ("95% TPR threshold"); one injected site | Threshold from clean calibration data only; faults sweep layers, neurons and all 32 bits |
| `abnormality_score3` compares `min_location` to `max_location` and returns 1 when both *match* (README says 1 when both *differ*) | tau3 as defined in the paper / README: 0, 0.5, 1 for 0, 1, 2 mismatches |
| `plt.hist`, per-value Python loops and `bisect` for histograms | Vectorized torch bucketing; no matplotlib in the compute path |
| Forward hooks registered on every call and never removed; captured `output.detach()` shares storage, so a following in-place ReLU can overwrite it (not verified by running) | Hooks via context manager; values read inside the hook |
| `pytorchfi` fault injector | Small in-process injector (`drdna.faults`) with an involution test |
| Pickled profiles and hand-edited absolute paths | Profiles are computed in memory, seeds are explicit |

The original scripts remain under `DRDNA/` and `src/` untouched for comparison. tau2 uses a
bin-normalized EMD (0-1) rather than activation units; this is a documented deviation.

## Limitations

* One model (CIFAR ResNet-18), one error model (single-bit neuron output flips), 19 monitored layers
  (conv, bn, avgpool), one seed per campaign. Only 39 / 139 SDC trials, hence the wide intervals.
* CPU float32 only; faults are applied to every sample in a batch at the same location.
* The paper's mitigation stage is not implemented.

---

# Original write-up


# Dr. DNA (unofficial reproduction)

An independent, from-scratch reproduction of one detection path from **"Dr. DNA: Combating Silent Data Corruptions in Deep Learning using Distribution of Neuron Activations"** (Ma, Lin, Desmaison, Coburn, Moore, Sankar, Jiao — **ASPLOS '24**, pp. 239-252). No official implementation was available at the time of writing.

> **Scope of this reproduction:** the paper evaluates 3 vision tasks, 5 datasets, 10 models, and 4 error models, plus a mitigation stage. This repo reproduces the **detection** path only, on **ResNet-18 / CIFAR-10** under a **single-bit-flip** error model. It is not a full reproduction of the paper's results.

```bibtex
@inproceedings{ma2024drdna,
  title     = {Dr. DNA: Combating Silent Data Corruptions in Deep Learning using Distribution of Neuron Activations},
  author    = {Ma, Dongning and Lin, Fred and Desmaison, Alban and Coburn, Joel and Moore, Daniel and Sankar, Sriram and Jiao, Xun},
  booktitle = {Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3 (ASPLOS '24)},
  pages     = {239--252},
  year      = {2024},
  doi       = {10.1145/3620666.3651349}
}
```

This is an unofficial, independent reproduction and is not affiliated with the original authors or their institutions. Licensed under MIT (see `LICENSE`); please cite the original paper above if you build on the method.

## Overview

DrDNA is a framework for detecting and mitigating soft errors in deep neural networks. Soft errors, such as silent data corruptions, pose a significant challenge to reliable deep learning model deployment, particularly in safety-critical applications. DrDNA identifies soft errors through model behavior and implements techniques to mitigate the impact of these errors.

## Methodology 

This is a post-hoc method and involves 2 major steps:
### Profiling - 
Each Layer in the model has k sites (cohort size)  as detection sites. 
these sites are used to generate there stats for the data Tau1,Tau2,Tau3

Tau1 - Each neuron in the k sites produces activation on inference. a histogram of activations is created for each such site in each layer

Tau2 - A histogram of activations of all the detection sites in a layer 

Tau3 - Neurons producing highest and lowest activations. 
### Scoring
These 3 profiles are used to generate 3 scores

Tau1 score - (1 - fi ) f1 is the frequency of the activation in the histogram of the site. (higher fi means higher frequency hence less abnormal) 

Tau2 score - EMD distance between layer histogram during inference and profiling . (high EMD means more abnormal)

Tau3 score - 0 if neuron are same , 0.5 if one neuron is same. 1 if both neurons are different

Score[layer] = lambda1 * tau1 + lambda2 * tau2 + lambda3 * tau3 + Score[layer-1]
###  Detection

The scores are generated for data without fault.

Since we have the layer-wise scores for data without fault during profiling we can use them for early detection. 

T` score for layer j (with fault)

T score for layer j (no fault, available after profiling) 

Margin = (T` - T)/ T

Margin can be configured experminatally. 
if Margin > thresold SDC Detected!
else not detected.

# Assumptions
Used hyperparameters Lambda1 ,Lambda2 ,Lambda3  = 1

##
Dataset = Cifar 10 

Model  = Resnet 18 

Used 1000 samples for profiling. 

# Detection Results
100% SDC detected

Successfully able to replicate the results in the paper. 

Fault injected using custom fault injector based on pytorchFI. 
# Abnormality scores analysis 


![alt text](https://github.com/amanyagami/DRDNA/blob/main/output.png))

The affect of SDC increases the abnormality score of the values. 
 95% TPR threshold = 499. 
