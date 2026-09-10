"""
Headless-browser validation of docs/index.html's live ONNX demo: load the
real page, run the real clean-inference button, and check the score against
the Python reference computed for the same example in export_browser_demo.py
(examples.json's clean_score). Also exercises the fault-injection button and
sanity-checks the badge / detected-vs-not logic responds to a real score.

Run from ~/drdna-browser (needs `python3 -m http.server` reachable, and
playwright installed).
"""
import json
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

PORT = 8791

server = subprocess.Popen(
    [sys.executable, "-m", "http.server", str(PORT), "--directory", "docs"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(1.0)

examples = json.load(open("docs/data/examples.json"))

failures = []

try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: console_errors.append(str(exc)))

        print(f"loading http://localhost:{PORT}/index.html ...")
        page.goto(f"http://localhost:{PORT}/index.html", timeout=60000)

        print("waiting for model load (status -> 'ready')...")
        page.wait_for_function(
            "document.getElementById('liveStatus').textContent.includes('ready')",
            timeout=120000,
        )
        print("model loaded:", page.eval_on_selector("#liveStatus", "el => el.textContent"))

        if console_errors:
            print("CONSOLE ERRORS DURING LOAD:", console_errors)
            failures.append(f"console errors during load: {console_errors}")

        # ---- selected example should be examples[0] by default ----
        label = page.eval_on_selector("#previewLabel", "el => el.textContent")
        expected_label = examples[0]["label"]
        print(f"default example label: '{label}' (expect '{expected_label}')")
        if label.strip() != expected_label:
            failures.append(f"default example label mismatch: got '{label}', expected '{expected_label}'")

        # ---- clean inference ----
        print("clicking 'Run clean inference' ...")
        page.click("#btnRunClean")
        page.wait_for_function(
            "document.getElementById('liveCleanScore').textContent !== '—'", timeout=30000
        )
        clean_score_js = float(page.eval_on_selector("#liveCleanScore", "el => el.textContent"))
        expected_clean = examples[0]["clean_score"]
        diff = abs(clean_score_js - expected_clean)
        print(f"JS clean score = {clean_score_js}, Python reference = {expected_clean:.2f}, diff = {diff:.4f}")
        if diff > 0.5:
            failures.append(f"clean score mismatch: JS={clean_score_js} python={expected_clean} diff={diff}")

        caption = page.eval_on_selector("#liveCaption", "el => el.textContent")
        print("caption:", caption)
        if expected_label not in caption and examples[0]["clean_pred"] not in caption:
            failures.append(f"caption doesn't mention predicted class: {caption}")

        # ---- fault injection (real, random site each time -- just check it
        #      runs, produces a real number, and the badge matches the
        #      threshold logic) ----
        print("clicking 'Inject a real fault & run' ...")
        page.click("#btnRunFault")
        page.wait_for_function(
            "document.getElementById('liveFaultScore').textContent !== '—'", timeout=30000
        )
        fault_score_js = float(page.eval_on_selector("#liveFaultScore", "el => el.textContent"))
        badge_text = page.eval_on_selector("#liveBadge", "el => el.textContent")
        fault_site = page.eval_on_selector("#liveFaultSite", "el => el.textContent")
        print(f"JS fault score = {fault_score_js}, badge = '{badge_text}'")
        print("fault site:", fault_site)

        detected = fault_score_js > 499
        if detected and "detected" not in badge_text.lower():
            failures.append(f"score {fault_score_js} > 499 but badge says '{badge_text}'")
        if not detected and "not detected" not in badge_text.lower():
            failures.append(f"score {fault_score_js} <= 499 but badge says '{badge_text}'")

        # sanity bounds -- a real Dr. DNA score on this setup should land
        # roughly in the 400s-500s+ range (see RESULTS.md); anything wildly
        # outside that suggests the JS port has a real bug, not "it's just a
        # different valid fault site".
        if not (300 < fault_score_js < 700):
            failures.append(f"fault score {fault_score_js} outside sane range (300,700)")

        if console_errors:
            failures.append(f"console errors after interactions: {console_errors}")

        # ---- switch example + rerun to make sure UI state updates cleanly ----
        print("switching to 2nd thumbnail...")
        page.click(".thumb:nth-child(2)")
        label2 = page.eval_on_selector("#previewLabel", "el => el.textContent")
        print("switched to:", label2, "(expect", examples[1]["label"], ")")
        if label2.strip() != examples[1]["label"]:
            failures.append(f"thumbnail switch didn't update label: got {label2}")
        score_reset = page.eval_on_selector("#liveCleanScore", "el => el.textContent")
        looks_numeric = any(ch.isdigit() for ch in score_reset)
        if looks_numeric:
            failures.append(f"clean score didn't reset on example switch (still numeric): {score_reset!r}")
        else:
            print(f"clean score correctly reset to non-numeric placeholder: {score_reset!r}")

        page.click("#btnRunClean")
        page.wait_for_function(
            "document.getElementById('liveCleanScore').textContent !== '—'", timeout=30000
        )
        clean2_js = float(page.eval_on_selector("#liveCleanScore", "el => el.textContent"))
        expected2 = examples[1]["clean_score"]
        diff2 = abs(clean2_js - expected2)
        print(f"2nd example: JS={clean2_js} python={expected2:.2f} diff={diff2:.4f}")
        if diff2 > 0.5:
            failures.append(f"2nd example clean score mismatch: JS={clean2_js} python={expected2} diff={diff2}")

        browser.close()
finally:
    server.terminate()

print("\n" + "=" * 60)
if failures:
    print(f"FAILED ({len(failures)} issue(s)):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
