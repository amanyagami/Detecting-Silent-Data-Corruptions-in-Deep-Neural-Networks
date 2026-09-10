import json

from playwright.sync_api import sync_playwright

cases = json.load(open("browser_export/reference_faults.json"))

FLIP_FN = """
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
"""

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("about:blank")
    ok = True
    for c in cases:
        js = "() => { " + FLIP_FN + f"return flipBit({c['orig_value']!r}, {c['bit_pos']}); }}"
        result = page.evaluate(js)
        expected = c["corrupted_value"]
        diff = abs(result - expected)
        match = diff < 1e-6
        print(
            f"bit_pos={c['bit_pos']} orig={c['orig_value']:.6f} -> "
            f"JS={result:.6f} python={expected:.6f} diff={diff:.2e} {'OK' if match else 'MISMATCH'}"
        )
        ok = ok and match
    browser.close()
    print("ALL BITFLIP CHECKS PASSED" if ok else "BITFLIP MISMATCH FOUND")
