"""Fault-injection campaign with a calibrated detection threshold.

Protocol (differences from the original scripts are intentional):

* Profiles are built on training images; the threshold is the ``1 - alpha`` quantile of the
  final score on a *separate* clean calibration set, never on faulty data.
* Faults are sampled over layers, neurons and all 32 bit positions, not a single site.
* Every trial is classified as ``masked`` (top-1 unchanged), ``sdc`` (top-1 changed, outputs
  finite) or ``nonfinite`` (NaN/Inf outputs), and detection is reported per class.
* The false-positive rate is measured on clean evaluation images that were not used to set
  the threshold.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from drdna.data import make_splits
from drdna.detector import DrDNA, default_layers
from drdna.faults import NeuronFault
from drdna.model import cifar_resnet18
from drdna.stats import auroc, wilson

# name -> (lambdas for tau1/tau2/tau3, components used). The cohort has 32 sites, so tau1
# (a sum over sites, 0..32) is divided by 32 in the balanced variant to match tau2 (0..1).
ABLATIONS = {
    "tau1": ((1.0, 1.0, 1.0), "1"),
    "tau2": ((1.0, 1.0, 1.0), "2"),
    "tau3": ((1.0, 1.0, 1.0), "3"),
    "tau1+2": ((1.0, 1.0, 1.0), "12"),
    "tau1/k+2": ((1 / 32, 1.0, 1.0), "12"),
    "all": ((1.0, 1.0, 1.0), "123"),
}


def bit_class(bit: int) -> str:
    """Name the float32 field a bit position belongs to."""
    if bit == 31:
        return "sign"
    if bit >= 23:
        return "exponent"
    return "mantissa-high" if bit >= 16 else "mantissa-low"


def _batches(x: torch.Tensor, size: int = 100):
    for i in range(0, len(x), size):
        yield x[i : i + size]


@torch.no_grad()
def clean_scores(det: DrDNA, model, x: torch.Tensor) -> dict[str, np.ndarray]:
    """Cumulative final-layer scores of clean inputs for every ablation."""
    out: dict[str, list[np.ndarray]] = {k: [] for k in ABLATIONS}
    for xb in _batches(x):
        _, s = det.score(model, xb)
        for name, (lam, mask) in ABLATIONS.items():
            out[name].append(s.cumulative(lam, mask)[:, -1].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def run(
    weights: str,
    cifar_root: str,
    n_faults: int = 300,
    imgs_per_fault: int = 16,
    alphas: tuple[float, ...] = (0.01, 0.05),
    seed: int = 0,
    bit_range: tuple[int, int] = (0, 31),
) -> dict:
    """Run the campaign and return a JSON-serializable result dictionary."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = cifar_resnet18(weights)
    splits = make_splits(cifar_root, seed=seed)
    layers = default_layers(model)
    det = DrDNA(layers, seed=seed).fit(model, _batches(splits.profile))

    with torch.no_grad():
        clean_pred = torch.cat([model(b).argmax(1) for b in _batches(splits.evaluation)])
    ok = (clean_pred == splits.evaluation_labels).nonzero().flatten()
    accuracy = len(ok) / len(clean_pred)
    pool = splits.evaluation[ok]  # only inputs the fault-free model gets right

    calib = clean_scores(det, model, splits.calib)
    held = clean_scores(det, model, pool)  # fault-free evaluation images (FPR measurement)
    thr = {a: {k: float(np.quantile(calib[k], 1 - a)) for k in ABLATIONS} for a in alphas}

    shapes = {n: det._shape[n] for n in layers}
    rows: list[dict] = []
    for _ in range(n_faults):
        layer = layers[int(rng.integers(len(layers)))]
        shp = shapes[layer]
        c = int(rng.integers(shp[0]))
        h = int(rng.integers(shp[1])) if len(shp) == 3 else 0
        w = int(rng.integers(shp[2])) if len(shp) == 3 else 0
        bit = int(rng.integers(bit_range[0], bit_range[1] + 1))
        sel = rng.choice(len(pool), size=imgs_per_fault, replace=False)
        x = pool[sel]
        with torch.no_grad():
            ref = model(x).argmax(1)
        with NeuronFault(model, layer, c, h, w, bit):
            logits, s = det.score(model, x)
        nonfinite = ~torch.isfinite(logits).all(1)
        changed = logits.argmax(1) != ref
        final = {k: s.cumulative(lam, m)[:, -1].numpy() for k, (lam, m) in ABLATIONS.items()}
        for i in range(imgs_per_fault):
            outcome = "nonfinite" if nonfinite[i] else ("sdc" if changed[i] else "masked")
            rows.append(
                {"layer": layer, "bit": bit, "outcome": outcome}
                | {k: float(v[i]) for k, v in final.items()}
            )

    def summarize(subset: list[dict], key: str, alpha: float) -> dict:
        n = len(subset)
        k = sum(r[key] > thr[alpha][key] for r in subset)
        lo, hi = wilson(k, n)
        return {"n": n, "detected": k, "rate": k / n if n else float("nan"), "ci95": [lo, hi]}

    result: dict = {
        "config": {
            "faults": n_faults,
            "images_per_fault": imgs_per_fault,
            "seed": seed,
            "layers": layers,
            "clean_accuracy": accuracy,
            "eval_images": len(pool),
        },
        "outcome_counts": {
            o: sum(r["outcome"] == o for r in rows) for o in ("masked", "sdc", "nonfinite")
        },
        "false_positive_rate": {},
        "detection": {},
        "auroc_clean_vs_sdc": {},
        "by_bit_class": {},
        "by_layer": {},
    }
    sdc = [r for r in rows if r["outcome"] == "sdc"]
    for key in ABLATIONS:
        neg = held[key]
        pos = np.array([r[key] for r in sdc])
        result["auroc_clean_vs_sdc"][key] = auroc(neg, pos) if len(pos) else None
        for a in alphas:
            fp = int((neg > thr[a][key]).sum())
            result["false_positive_rate"][f"{key}@{a}"] = {
                "fp": fp,
                "n": len(neg),
                "rate": fp / len(neg),
                "ci95": list(wilson(fp, len(neg))),
            }
            for outcome in ("sdc", "masked", "nonfinite"):
                sub = [r for r in rows if r["outcome"] == outcome]
                result["detection"][f"{key}@{a}/{outcome}"] = summarize(sub, key, a)
    a0 = alphas[0]
    for bc in ("sign", "exponent", "mantissa-high", "mantissa-low"):
        sub = [r for r in sdc if bit_class(r["bit"]) == bc]
        result["by_bit_class"][bc] = summarize(sub, "all", a0) | {
            "all_trials": sum(bit_class(r["bit"]) == bc for r in rows)
        }
    for ly in layers:
        sub = [r for r in sdc if r["layer"] == ly]
        result["by_layer"][ly] = summarize(sub, "all", a0)
    return result


def report(res: dict, alpha: float) -> str:
    """Render the key results as markdown."""
    cfg, oc = res["config"], res["outcome_counts"]
    total = sum(oc.values())
    lines = [
        f"Clean accuracy {cfg['clean_accuracy']:.2%}; {cfg['faults']} random faults x "
        f"{cfg['images_per_fault']} images = {total} trials "
        f"(masked {oc['masked']}, sdc {oc['sdc']}, nonfinite {oc['nonfinite']}).",
        "",
        f"| detector | FPR@{alpha} (clean) | TPR on SDC | AUROC clean vs SDC |",
        "|---|---|---|---|",
    ]
    for k in ABLATIONS:
        fpr = res["false_positive_rate"][f"{k}@{alpha}"]
        d = res["detection"][f"{k}@{alpha}/sdc"]
        au = res["auroc_clean_vs_sdc"][k]
        lines.append(
            f"| {k} | {fpr['rate']:.3f} ({fpr['fp']}/{fpr['n']}) | "
            f"{d['rate']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] (n={d['n']}) | "
            f"{'n/a' if au is None else f'{au:.3f}'} |"
        )
    lines += [
        "",
        f"SDC detection (all components, FPR {alpha}) by flipped field:",
        "",
        "| field | SDC trials | detected | rate |",
        "|---|---|---|---|",
    ]
    for bc, d in res["by_bit_class"].items():
        lines.append(f"| {bc} | {d['n']} | {d['detected']} | {d['rate']:.3f} |")
    m = res["detection"][f"all@{alpha}/masked"]
    lines += ["", f"Masked flips flagged: {m['detected']}/{m['n']} (harmless to the output)."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``drdna-eval --weights ... --cifar-root ...``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="state_dicts/resnet18.pt")
    ap.add_argument("--cifar-root", default="data")
    ap.add_argument("--faults", type=int, default=300)
    ap.add_argument("--images-per-fault", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bit-min", type=int, default=0, help="lowest bit to flip (0-31)")
    ap.add_argument("--bit-max", type=int, default=31, help="highest bit to flip (0-31)")
    ap.add_argument("--tag", default="", help="suffix for the result file names")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    t0 = time.time()
    res = run(
        args.weights,
        args.cifar_root,
        args.faults,
        args.images_per_fault,
        seed=args.seed,
        bit_range=(args.bit_min, args.bit_max),
    )
    res["config"]["bit_range"] = [args.bit_min, args.bit_max]
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / f"campaign_seed{args.seed}{args.tag}.json").write_text(json.dumps(res, indent=1))
    text = report(res, 0.01)
    (out / f"campaign_seed{args.seed}{args.tag}.md").write_text(text + "\n")
    print(text)
    print(f"\n({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
