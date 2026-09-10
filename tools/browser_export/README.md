# Browser export & validation

Produces and verifies the assets behind the live in-browser demo on the
[project page](../../docs/index.html) (`docs/models/*.onnx`,
`docs/data/*.json`) — real ONNX inference + a real single-bit IEEE-754
fault, not a simulation. See the page's "run it live" section for context.

- **`export_browser_demo.py`** — loads the real trained ResNet-18, splits it
  into two ONNX graphs at the `layer1`→`layer2` boundary (so JS can inject a
  real fault into the real intermediate tensor between them), exports the
  real tau1/tau2/tau3 profile to JSON, and exports 8 real CIFAR-10 test
  images with their real reference clean scores. Cross-checks the ONNX
  outputs against the original PyTorch model at export time.
- **`reference_fault_check.py`** — generates ground-truth *faulted* scores
  (real bit-flip, real corrupted forward pass, real PyTorch model) for a
  few fixed (image, neuron, bit) combinations, used to validate the
  browser's JS port below.
- **`check_bitflip.py`** — headless-browser check that the page's JS
  IEEE-754 bit-flip function produces bit-identical output to Python's, for
  the exact cases `reference_fault_check.py` generated.
- **`validate_browser.py`** — headless-browser (Playwright) end-to-end
  check: loads the real page, clicks the real buttons, and asserts the
  JS-computed clean scores match `export_browser_demo.py`'s Python
  reference scores to float precision, across multiple images.

Run from the repo root, in a venv with `torch`, `onnx`, `onnxruntime`,
`scipy`, `pandas`, `numpy`, and (for the two validation scripts)
`playwright` (`playwright install chromium`):

```bash
python3 tools/browser_export/export_browser_demo.py
python3 tools/browser_export/reference_fault_check.py
python3 tools/browser_export/check_bitflip.py
python3 tools/browser_export/validate_browser.py
```

`export_browser_demo.py` writes into `browser_export/` at the repo root;
copy `part_a.onnx` / `part_b.onnx` into `docs/models/` and `profile.json` /
`examples.json` into `docs/data/` to update the live page.

**On Git LFS:** intentionally *not* used for `docs/models/*.onnx` even
though this repo uses LFS elsewhere — GitHub Pages serves LFS-tracked files
as their raw pointer text, not the real binary, which would silently break
the live demo.
