#!/usr/bin/env python3
"""Drive the web app with Playwright and save screenshots at desktop and phone width.

    python scripts/screenshots.py            # all flows
    python scripts/screenshots.py --only okota

Needs both servers running, the backend with FLOODLENS_DEMO_CACHE=1 so the two known
sites replay instantly. Install with:  pip install playwright && playwright install chromium
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PWTimeout

WEB = os.environ.get("FLOODLENS_WEB", "http://localhost:3000")
SHOTS = Path(os.environ.get("FLOODLENS_SHOTS",
                            Path(__file__).resolve().parents[2] / "web" / "screenshots"))
DESKTOP = {"width": 1440, "height": 1000}
PHONE = {"width": 390, "height": 844}  # iPhone 14 logical width
PHOTOS = Path.home() / "Downloads" / "site_photos" / "jpg"

SITES = {
    "okota": {"lat": "6.5026016", "lon": "3.3051669", "address": "24 Ago Palace Way, Okota",
              "label": "Okota shop", "amount": "1500000", "tenor": "12"},
    "lekki": {"lat": "6.4413987", "lon": "3.4704698", "address": "Admiralty Way, Lekki Phase 1",
              "label": "Lekki Phase 1 shop", "amount": "2400000", "tenor": "18"},
    "fresh": {"lat": "6.6018", "lon": "3.3515", "address": "Oke-Ira, Ogba",
              "label": "Ogba lock-up", "amount": "800000", "tenor": "9"},
}


def shot(page: Page, name: str) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    path = SHOTS / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    print(f"    saved {path.relative_to(SHOTS.parents[1])}")


def fill_form(page: Page, site: dict, photos: list[Path] | None = None) -> None:
    page.goto(f"{WEB}/?lat={site['lat']}&lon={site['lon']}", wait_until="networkidle")
    page.wait_for_selector("canvas.maplibregl-canvas", timeout=30_000)
    page.wait_for_timeout(1500)  # let the tiles paint before the screenshot
    page.fill('input[name="address"]', site["address"])
    page.fill('input[name="label"]', site["label"])
    page.fill('input[name="loan_amount"]', site["amount"])
    page.fill('input[name="tenor_months"]', site["tenor"])
    page.fill('textarea[name="note"]', "Ground floor shop, concrete slab, drain at the back")
    if photos:
        page.set_input_files('input[type="file"]', [str(p) for p in photos])
        page.wait_for_timeout(300)


def run_flow(page: Page, key: str, width_tag: str, photos: list[Path] | None = None,
             catch_progress: bool = False) -> str:
    site = SITES[key]
    print(f"  {key} @ {width_tag}{' with photos' if photos else ''}")
    fill_form(page, site, photos)
    shot(page, f"{key}-1-new-assessment-{width_tag}")

    page.click('button[type="submit"]')
    page.wait_for_url("**/assessments/**", timeout=30_000)
    assessment_id = page.url.rstrip("/").split("/")[-1]

    if catch_progress:
        try:  # a cached site finishes before we can see this
            page.wait_for_selector("text=Working on it", timeout=4_000)
            page.wait_for_timeout(2_500)
            shot(page, f"{key}-2-agent-at-work-{width_tag}")
        except PWTimeout:
            print("    (finished too fast to catch the progress screen)")

    page.wait_for_selector("text=What drives this score", timeout=600_000)
    page.wait_for_timeout(1_200)
    shot(page, f"{key}-3-risk-report-{width_tag}")
    return assessment_id


def check(page: Page, label: str, ok: bool) -> bool:
    print(f"    {'PASS' if ok else 'FAIL'}  {label}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SITES) + ["portfolio"], action="append")
    args = ap.parse_args()
    wanted = args.only or ["okota", "lekki", "fresh", "portfolio"]
    photos = sorted(PHOTOS.glob("*.jpg"))[:3]
    passed = True

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for tag, size in (("1440", DESKTOP), ("390", PHONE)):
            context = browser.new_context(viewport=size, device_scale_factor=2)
            page = context.new_page()
            page.on("pageerror", lambda e: print(f"    PAGE ERROR: {e}"))
            page.on("console", lambda m: print(f"    console.{m.type}: {m.text}")
                    if m.type == "error" else None)

            if "okota" in wanted:
                run_flow(page, "okota", tag)
                body = page.inner_text("body")
                passed &= check(page, "Okota: tier High", "High" in body)
                passed &= check(page, "Okota: verified badge",
                                "every number matches a measured value" in body)
                passed &= check(page, "Okota: flood report cards", "Read the source" in body)
                passed &= check(page, "Okota: weighted score and floor both shown",
                                "held at High by the evidence rule" in body
                                and "Weighted score" in body)
                passed &= check(page, "Okota: reports grouped by event",
                                "2 reports" in body and "1 report" in body)
                limits = body.split("What this report cannot tell you")[-1]
                passed &= check(page, "Okota: one height limit only",
                                limits.lower().count("height above") <= 1)

            if "lekki" in wanted:
                run_flow(page, "lekki", tag)
                body = page.inner_text("body")
                passed &= check(page, "Lekki: no-reports sentence",
                                "No street-level flood reports found. This does not mean the "
                                "site is safe." in body)
                passed &= check(page, "Lekki: no missing-evidence banner",
                                "kind of evidence is missing" not in body
                                and "Some evidence is missing" not in body)
                limits = body.split("What this report cannot tell you")[-1]
                passed &= check(page, "Lekki: one news-coverage limit only",
                                sum(1 for w in ("news", "coverage") if w in limits.lower())
                                and limits.lower().count("covers some") <= 1)
                passed &= check(page, "Lekki: no duplicate limit bullets",
                                body.count("This does not mean the site is safe.") <= 2)

            if "fresh" in wanted:
                run_flow(page, "fresh", tag, photos=photos, catch_progress=True)
                body = page.inner_text("body")
                passed &= check(page, "fresh: photos shown", "Site photos" in body)
                passed &= check(page, "fresh: photos do not score",
                                "Nothing here changes the score" in body)

            if "portfolio" in wanted:
                page.goto(f"{WEB}/portfolio", wait_until="networkidle")
                shot(page, f"portfolio-{tag}")

            context.close()
        browser.close()
    print("\nALL CHECKS PASSED" if passed else "\nSOME CHECKS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
