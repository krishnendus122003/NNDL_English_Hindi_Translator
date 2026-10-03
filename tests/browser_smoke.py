"""Optional real-browser verification. Core unittest discovery does not import this.

Install Playwright into .test-tools, then run: python tests/browser_smoke.py
Requires local Edge and both demo servers; no browser download or user profile used.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".test-tools"))
from playwright.sync_api import sync_playwright, expect


def main():
    checks, errors = [], []
    def passed(name):
        checks.append(name)
        print("PASS", name)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1100})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:5500", wait_until="networkidle")
        expect(page.locator("#connection")).to_contain_text("API connected")
        passed("frontend opens and reaches API")
        expect(page.locator("#model")).to_have_value("transformer")
        expect(page.locator("#decoding")).to_have_value("beam")
        passed("default scratch Transformer and beam selected")

        def run(button="Translate"):
            page.get_by_role("button", name=button, exact=True).click()
            expect(page.locator("#translate")).to_be_enabled(timeout=120000)

        for sentence in ("I am happy.", "How are you?", "India is a diverse country."):
            page.locator("#text").fill(sentence)
            for model, decoding in (("gru", "greedy"), ("attention", "greedy"), ("transformer", "greedy"), ("transformer", "beam")):
                page.locator("#model").select_option(model)
                if model == "transformer":
                    page.locator("#decoding").select_option(decoding)
                run()
                expect(page.locator("#error")).to_be_hidden()
                expect(page.locator("#metadata")).to_contain_text(decoding)
                assert page.locator("#output").inner_text().strip()
                passed(f"{model} {decoding}: {sentence}")
        page.locator("#text").fill("I am happy.")
        run("Show Attention")
        expect(page.locator("#alignment")).to_be_visible()
        expect(page.locator("#warning")).to_contain_text("GRU + Bahdanau Attention")
        assert page.locator("#heatmap tbody td").count() > 0
        assert page.locator("#heatmap tbody th").count() == page.locator("#heatmap tbody tr").count()
        passed("genuine attention heatmap visible with model attribution")
        page.screenshot(path=str(ROOT / "docs/ui_attention.png"), full_page=True)
        run("Compare Models")
        assert page.locator(".compare-item").count() == 3
        passed("core comparison renders three models")
        page.locator("#include-nllb").check()
        run("Compare Models")
        assert page.locator(".compare-item").count() == 3
        expect(page.locator("#compare-results .error")).to_contain_text("unavailable")
        passed("optional NLLB unavailable preserves core comparison")
        page.locator("#include-nllb").uncheck()
        page.locator("#text").fill("I am happy. " * 80)
        page.locator("#model").select_option("gru")
        run()
        expect(page.locator("#warning")).to_contain_text("truncated")
        passed("long input warning visible")
        page.get_by_role("button", name="Clear", exact=True).click()
        run()
        expect(page.locator("#error")).to_contain_text("Enter an English sentence")
        passed("clear and empty input validation")
        page.get_by_role("button", name="I am happy.", exact=True).click()
        expect(page.locator("#text")).to_have_value("I am happy.")
        passed("example buttons populate input")
        page.locator("#model").select_option("transformer")
        page.locator("#decoding").select_option("beam")
        run()
        context.grant_permissions(["clipboard-read", "clipboard-write"])
        page.get_by_role("button", name="Copy Translation", exact=True).click()
        expect(page.locator("#copy")).to_have_text("Copied")
        assert page.evaluate("navigator.clipboard.readText()") == page.locator("#output").inner_text()
        passed("copy translation writes clipboard")
        page.screenshot(path=str(ROOT / "docs/ui_desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(ROOT / "docs/ui_mobile.png"), full_page=True)
        passed("mobile layout has no horizontal page overflow")
        page.route("**/translate", lambda route: route.fulfill(status=503, content_type="application/json", body='{"detail":"Test model unavailable"}'))
        run()
        expect(page.locator("#error")).to_contain_text("Test model unavailable")
        passed("API error displayed and controls recover")
        assert not errors, errors
        passed("no uncaught browser JavaScript errors")
        browser.close()
    report = {"passed": len(checks), "failed": 0, "checks": checks, "javascript_errors": errors,
              "browser": "Local Microsoft Edge, isolated headless context",
              "note": "In-app browser initialization failed with Windows sandbox error 5; approved local browser fallback used."}
    (ROOT / "docs/browser_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Browser checks: {len(checks)} passed, 0 failed")


if __name__ == "__main__":
    main()
