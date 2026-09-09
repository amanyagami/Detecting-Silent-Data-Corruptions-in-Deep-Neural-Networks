
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

See `RESULTS.md` for the broader evaluation (a second dataset and a second error model) and how it compares to the numbers above and to the original paper.

## Development

Install in editable mode with test dependencies:

```bash
pip install -e ".[dev]"
git lfs pull   # fetches state_dicts/, correct_data*.pt, tau*.pkl
```

Run the test suite:

```bash
pytest -v
```

`tests/` is a real pytest suite (unit tests for the bit-flip and tau-scoring
math, plus an end-to-end regression test that the SDC/clean score separation
holds). It runs in CI (`.github/workflows/test.yml`) on every push and PR.
The original loose experiment scripts (`resnet_fi.py`, etc.) still exist
under `experiments/` for manual/interactive use, but aren't part of the
automated suite.

The reusable pipeline itself lives in `src/detection/pipeline.py`
(`Profile`, `evaluate`, `register_hooks`, `score_sample`) and
`src/faults/` (the PyTorchFI-based injectors) -- both scripts and the tests
import from there rather than duplicating the scoring loop.
