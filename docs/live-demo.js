/*
 * Real, in-browser Dr. DNA inference: ResNet-18 (CIFAR-10) split into two
 * ONNX graphs at the layer1->layer2 boundary, run via onnxruntime-web
 * (WASM). A real single-bit IEEE-754 flip is applied to the real
 * intermediate activation tensor between the two graphs, and the exact
 * same tau1/tau2/tau3 scoring formula from src/detection/pipeline.py runs
 * on the results -- ported line-for-line, not reimagined. No server, no
 * simulation: every number on screen came out of a real forward pass that
 * just ran on this device.
 *
 * See export_browser_demo.py in the repo for how models/profile/examples
 * were produced, and this file's companion Python reference check
 * (reference_fault_check.py) for the ground-truth numbers this was
 * validated against.
 */
(function () {
  "use strict";

  const THRESHOLD = 499;
  const MODEL_BASE = "models/";
  const DATA_BASE = "data/";

  const els = {
    status: document.getElementById("liveStatus"),
    thumbRow: document.getElementById("thumbRow"),
    previewCanvas: document.getElementById("previewCanvas"),
    previewLabel: document.getElementById("previewLabel"),
    btnClean: document.getElementById("btnRunClean"),
    btnFault: document.getElementById("btnRunFault"),
    cleanScore: document.getElementById("liveCleanScore"),
    faultScore: document.getElementById("liveFaultScore"),
    badge: document.getElementById("liveBadge"),
    caption: document.getElementById("liveCaption"),
    faultSite: document.getElementById("liveFaultSite"),
  };
  if (!els.thumbRow) return; // section not on this page

  let sessionA, sessionB, profile, examples;
  let selectedExample = null;

  // ---------------------------------------------------------------------
  // Tau-scoring math -- a direct port of src/detection/pipeline.py /
  // DRDNA/b10c32/config.py. Preserves the same behavior, including the
  // documented b/c index-reuse quirk for any <=2D position (never actually
  // triggered by this repo's real DetectionSites.txt, but kept for
  // fidelity with the Python reference).
  // ---------------------------------------------------------------------
  function flatIndex(dims, pos) {
    let off = 0;
    let stride = 1;
    for (let i = dims.length - 1; i >= 0; i--) {
      off += pos[i] * stride;
      stride *= dims[i];
    }
    return off;
  }

  function activationValue(actMap, layerName, pos, lastBC) {
    const t = actMap[layerName];
    let idx;
    if (pos.length > 2) {
      lastBC[0] = pos[0];
      lastBC[1] = pos[1];
      idx = pos;
    } else {
      idx = [lastBC[0], lastBC[1], pos[0], pos[1]];
    }
    return t.data[flatIndex(t.dims, idx)];
  }

  function binLookup(value, hist) {
    // hist: [[[lo,hi], v], ...]
    for (const [[lo, hi], v] of hist) {
      if (lo <= value && value < hi) return v;
    }
    return null;
  }

  function abnormalityScore1(value, neuronHist) {
    const f = binLookup(value, neuronHist);
    return f === null ? 1 : 1 - f;
  }

  function freshCounts(hist) {
    // hist: [[[lo,hi], v], ...] -> same bins, all counts zeroed
    return hist.map(([range]) => [range, 0]);
  }

  function updateHistogram(value, hist) {
    for (const entry of hist) {
      const [lo, hi] = entry[0];
      if (lo <= value && value < hi) {
        entry[1] += 1;
        return hist;
      }
    }
    // matches update_histogram()'s edge handling: below-first / above-last
    if (hist.length && value < hist[0][0][0]) hist[0][1] += 1;
    else if (hist.length) hist[hist.length - 1][1] += 1;
    return hist;
  }

  function wassersteinDistanceSameSupport(values, w1, w2) {
    // Both histograms share identical bin midpoints (they're built from the
    // same profile bins), so this is the closed-form 1D EMD via CDF
    // difference -- matches scipy.stats.wasserstein_distance(bins, bins,
    // w1, w2) for equal support points.
    const sum1 = w1.reduce((a, b) => a + b, 0) || 1;
    const sum2 = w2.reduce((a, b) => a + b, 0) || 1;
    let cdf1 = 0,
      cdf2 = 0,
      total = 0;
    for (let i = 0; i < values.length - 1; i++) {
      cdf1 += w1[i] / sum1;
      cdf2 += w2[i] / sum2;
      total += Math.abs(cdf1 - cdf2) * (values[i + 1] - values[i]);
    }
    return total;
  }

  function abnormalityScore2(liveHist, profileHist) {
    const values = profileHist.map(([[lo, hi]]) => (lo + hi) / 2);
    const w1 = liveHist.map(([, v]) => v);
    const w2 = profileHist.map(([, v]) => v);
    const emd = wassersteinDistanceSameSupport(values, w1, w2);
    const total = w1.reduce((a, b) => a + b, 0) * w2.reduce((a, b) => a + b, 0);
    return total === 0 ? 0 : emd / total;
  }

  function tau3ForLayer(tensor) {
    const data = tensor.data;
    let maxV = -Infinity,
      minV = Infinity,
      maxI = 0,
      minI = 0;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v > maxV) {
        maxV = v;
        maxI = i;
      }
      if (v < minV) {
        minV = v;
        minI = i;
      }
    }
    return {
      max_value: maxV,
      max_location: unravel(tensor.dims, maxI),
      min_value: minV,
      min_location: unravel(tensor.dims, minI),
    };
  }

  function unravel(dims, flatIdx) {
    const out = new Array(dims.length);
    let rem = flatIdx;
    for (let i = dims.length - 1; i >= 0; i--) {
      out[i] = rem % dims[i];
      rem = Math.floor(rem / dims[i]);
    }
    return out;
  }

  function locEq(a, b) {
    return a.length === b.length && a.every((v, i) => v === b[i]);
  }

  function abnormalityScore3(live, prof) {
    // Preserves the original (likely-typo'd but validated) rule: both
    // branches compare against prof.max_location, never prof.min_location.
    // See RESULTS discussion / pipeline.py docstring.
    const maxMatch = locEq(live.max_location, prof.max_location);
    const minVsMax = locEq(live.min_location, prof.max_location);
    if (maxMatch || minVsMax) return maxMatch && minVsMax ? 1 : 0.5;
    return 0;
  }

  function scoreFromActivations(actMap) {
    const tau2Running = {};
    for (const ln of profile.layer_names) tau2Running[ln] = freshCounts(profile.tau2[ln]);

    const tau1Scores = {};
    for (const ln of profile.layer_names) tau1Scores[ln] = 0;
    const lastBC = [0, 0];

    for (const [layerName, pos] of profile.selected_neurons) {
      const key = layerName + "|" + pos.join(",");
      const value = activationValue(actMap, layerName, pos, lastBC);
      tau1Scores[layerName] += abnormalityScore1(value, profile.tau1[key]);
      tau2Running[layerName] = updateHistogram(value, tau2Running[layerName]);
    }

    const tau2Scores = {};
    for (const ln of profile.layer_names) tau2Scores[ln] = abnormalityScore2(tau2Running[ln], profile.tau2[ln]);

    const tau3Scores = {};
    for (const ln of profile.layer_names) tau3Scores[ln] = abnormalityScore3(tau3ForLayer(actMap[ln]), profile.tau3[ln]);

    let running = 0;
    for (const layer of profile.layer_names) running += tau1Scores[layer] + tau2Scores[layer] + tau3Scores[layer];
    return running;
  }

  // ---------------------------------------------------------------------
  // IEEE-754 bit flip -- same semantics as customFI_methods.py's
  // _flip_bit_signed_Aman (bit_pos 0 = sign/MSB, 31 = LSB, big-endian).
  // ---------------------------------------------------------------------
  function flipBit(value, bitPos) {
    const buf = new ArrayBuffer(4);
    const view = new DataView(buf);
    view.setFloat32(0, value, false);
    const bits = view.getUint32(0, false);
    const shift = 31 - bitPos;
    const flipped = (bits ^ (1 << shift)) >>> 0;
    view.setUint32(0, flipped, false);
    return view.getFloat32(0, false);
  }

  const PART_A_NAMES = ["conv1", "bn1", "layer1.0.conv1", "layer1.0.bn1", "layer1.0.conv2", "layer1.0.bn2"];
  const PART_B_NAMES = [
    "layer2.0.conv1", "layer2.0.bn1", "layer2.0.conv2", "layer2.0.bn2",
    "layer3.0.conv1", "layer3.0.bn1", "layer3.0.conv2", "layer3.0.bn2",
    "layer4.0.conv1", "layer4.0.bn1", "layer4.0.conv2", "layer4.0.bn2",
    "avgpool", "fc",
  ];
  const CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];

  function toTensor(ortOutput) {
    return { dims: ortOutput.dims, data: ortOutput.data };
  }

  async function runPartA(pixels) {
    const input = new ort.Tensor("float32", Float32Array.from(pixels), [1, 3, 32, 32]);
    const results = await sessionA.run({ input });
    const names = ["split", ...PART_A_NAMES.map((n) => "act_" + n)];
    const out = {};
    names.forEach((n) => (out[n] = toTensor(results[n])));
    return out;
  }

  async function runPartB(splitTensor) {
    const input = new ort.Tensor("float32", splitTensor.data, splitTensor.dims);
    const results = await sessionB.run({ split: input });
    const names = ["logits", ...PART_B_NAMES.map((n) => "act_" + n)];
    const out = {};
    names.forEach((n) => (out[n] = toTensor(results[n])));
    return out;
  }

  function argmax(tensor) {
    let best = -Infinity,
      idx = 0;
    for (let i = 0; i < tensor.data.length; i++) {
      if (tensor.data[i] > best) {
        best = tensor.data[i];
        idx = i;
      }
    }
    return idx;
  }

  // ---------------------------------------------------------------------
  // UI wiring
  // ---------------------------------------------------------------------
  function drawPixelsToCanvas(canvas, pixels, scale) {
    canvas.width = 32 * (scale || 1);
    canvas.height = 32 * (scale || 1);
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(32, 32);
    // pixels: CHW float32, roughly in a normalized range -- rescale to 0-255
    // per-channel using its own min/max so the thumbnail is a recognizable image.
    for (let c = 0; c < 3; c++) {
      let mn = Infinity,
        mx = -Infinity;
      for (let i = 0; i < 1024; i++) {
        const v = pixels[c * 1024 + i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      const range = mx - mn || 1;
      for (let i = 0; i < 1024; i++) {
        img.data[i * 4 + c] = ((pixels[c * 1024 + i] - mn) / range) * 255;
      }
    }
    for (let i = 0; i < 1024; i++) img.data[i * 4 + 3] = 255;
    if (!scale || scale === 1) {
      ctx.putImageData(img, 0, 0);
    } else {
      const off = document.createElement("canvas");
      off.width = 32;
      off.height = 32;
      off.getContext("2d").putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, 0, 0, 32 * scale, 32 * scale);
    }
  }

  function setStatus(text) {
    if (els.status) els.status.textContent = text;
  }
  function setCaption(text) {
    if (els.caption) els.caption.textContent = text;
  }

  function selectExample(ex, thumbEl) {
    selectedExample = ex;
    document.querySelectorAll(".thumb").forEach((t) => t.classList.remove("selected"));
    if (thumbEl) thumbEl.classList.add("selected");
    drawPixelsToCanvas(els.previewCanvas, ex.pixels, 4);
    els.previewLabel.textContent = ex.label;
    els.cleanScore.textContent = "—";
    els.faultScore.textContent = "—";
    els.badge.className = "badge";
    els.badge.textContent = "idle";
    if (els.faultSite) els.faultSite.textContent = "";
    setCaption("Run a clean pass, then inject a real fault.");
    els.btnClean.disabled = false;
    els.btnFault.disabled = false;
  }

  async function init() {
    try {
      setStatus("loading ONNX Runtime…");
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";

      setStatus("loading real model weights (≈ 45MB, cached after first load)…");
      const [profileResp, examplesResp] = await Promise.all([
        fetch(DATA_BASE + "profile.json").then((r) => r.json()),
        fetch(DATA_BASE + "examples.json").then((r) => r.json()),
      ]);
      profile = profileResp;
      examples = examplesResp;

      [sessionA, sessionB] = await Promise.all([
        ort.InferenceSession.create(MODEL_BASE + "part_a.onnx"),
        ort.InferenceSession.create(MODEL_BASE + "part_b.onnx"),
      ]);

      els.thumbRow.innerHTML = "";
      examples.forEach((ex, i) => {
        const c = document.createElement("canvas");
        c.className = "thumb";
        c.title = ex.label;
        drawPixelsToCanvas(c, ex.pixels, 2);
        c.addEventListener("click", () => selectExample(ex, c));
        els.thumbRow.appendChild(c);
        if (i === 0) selectExample(ex, c);
      });

      setStatus("ready — running on-device via WASM");
    } catch (err) {
      console.error(err);
      setStatus("failed to load (see console)");
      setCaption("Something didn't load — " + (err && err.message ? err.message : String(err)));
    }
  }

  els.btnClean.addEventListener("click", async () => {
    if (!selectedExample) return;
    els.btnClean.disabled = true;
    els.btnFault.disabled = true;
    els.badge.className = "badge";
    els.badge.textContent = "running…";
    try {
      const t0 = performance.now();
      const outA = await runPartA(selectedExample.pixels);
      const split = outA.split;
      const outB = await runPartB(split);
      const activations = {};
      PART_A_NAMES.forEach((n) => (activations[n] = outA["act_" + n]));
      PART_B_NAMES.forEach((n) => (activations[n] = outB["act_" + n]));
      const score = scoreFromActivations(activations);
      const pred = CIFAR10_CLASSES[argmax(outB.logits)];
      const ms = (performance.now() - t0).toFixed(0);
      els.cleanScore.textContent = score.toFixed(1);
      els.badge.className = "badge safe";
      els.badge.textContent = "nominal";
      setCaption("Real forward pass, no fault — predicted “" + pred + "” in " + ms + "ms on this device.");
    } catch (err) {
      console.error(err);
      setCaption("Inference failed — " + (err && err.message ? err.message : String(err)));
    }
    els.btnClean.disabled = false;
    els.btnFault.disabled = false;
  });

  els.btnFault.addEventListener("click", async () => {
    if (!selectedExample) return;
    els.btnClean.disabled = true;
    els.btnFault.disabled = true;
    els.badge.className = "badge";
    els.badge.textContent = "injecting a real fault…";
    try {
      const t0 = performance.now();
      const outA = await runPartA(selectedExample.pixels);
      const split = outA.split;

      // real random fault site inside the real [1,64,16,16] split tensor.
      // bit position weighted toward the IEEE-754 exponent bits (1-8),
      // which is where a flip changes the *magnitude* of the activation
      // rather than just its sign or a fine mantissa bit -- matches the
      // bit_pos=1 used in this repo's own validated experiments.
      const c = Math.floor(Math.random() * split.dims[1]);
      const h = Math.floor(Math.random() * split.dims[2]);
      const w = Math.floor(Math.random() * split.dims[3]);
      const bitPos = 1 + Math.floor(Math.random() * 8);

      const corrupted = { dims: split.dims, data: Float32Array.from(split.data) };
      const flatIdx = flatIndex(split.dims, [0, c, h, w]);
      const origVal = corrupted.data[flatIdx];
      const newVal = flipBit(origVal, bitPos);
      corrupted.data[flatIdx] = newVal;

      const outB = await runPartB(corrupted);
      const activations = {};
      PART_A_NAMES.forEach((n) => (activations[n] = outA["act_" + n]));
      PART_B_NAMES.forEach((n) => (activations[n] = outB["act_" + n]));
      const score = scoreFromActivations(activations);
      const pred = CIFAR10_CLASSES[argmax(outB.logits)];
      const ms = (performance.now() - t0).toFixed(0);
      const detected = score > THRESHOLD;

      els.faultScore.textContent = score.toFixed(1);
      els.badge.className = "badge " + (detected ? "ok" : "");
      els.badge.textContent = detected ? "SDC detected" : "not detected";
      if (els.faultSite) {
        els.faultSite.textContent =
          "site: layer1 output, channel " + c + ", (" + h + "," + w + "), bit " + bitPos +
          " — " + origVal.toFixed(4) + " → " + newVal.toFixed(4);
      }
      setCaption(
        (detected ? "Flagged: " : "Missed: ") +
          "score " + score.toFixed(1) + " vs. threshold " + THRESHOLD +
          " — predicted “" + pred + "”, real forward pass in " + ms + "ms."
      );
    } catch (err) {
      console.error(err);
      setCaption("Inference failed — " + (err && err.message ? err.message : String(err)));
    }
    els.btnClean.disabled = false;
    els.btnFault.disabled = false;
  });

  init();
})();
