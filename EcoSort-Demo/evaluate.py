"""Unified rule checks and refactor regression, not a new independent test."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import unquote, urlparse
import urllib.request
from zoneinfo import ZoneInfo

import gradio as gr
import numpy as np
from PIL import Image
from app import BASE, DISPLAY, EcoSort
from disposal_rules import ITEM_TYPES, disposal_guidance, predicted_item_guidance


def rule_checks():
    assert "yellow" in predicted_item_guidance("plastic", "plastic drink bottle")[0]
    assert "red" in predicted_item_guidance("plastic", "plastic bag")[0]
    assert "Potentially" in predicted_item_guidance("plastic", "plastic household item")[0]
    assert "red" in predicted_item_guidance("trash", "used tissue")[0]
    assert "drop-off" in predicted_item_guidance("battery", "household battery")[0]
    assert "Manual review" in predicted_item_guidance("trash", "used tissue", uncertain=True)[0]
    assert "User-confirmed" not in predicted_item_guidance("plastic", "plastic drink bottle")[1]
    assert "Confirm" in predicted_item_guidance("plastic", "plastic food container")[1]
    assert "otherwise red" in predicted_item_guidance("food_organics", "fruit peel")[0]
    assert "maroon" in predicted_item_guidance("food_organics", "fruit peel", True)[1]


def restore(item, path):
    parsed = urlparse(item["source_url"])
    assert parsed.scheme == "https" and parsed.netloc == "commons.wikimedia.org"
    title = unquote(parsed.path.removeprefix("/wiki/"))
    assert title.startswith("File:")
    from urllib.parse import urlencode
    params = urlencode(dict(action="query", titles=title, prop="imageinfo", iiprop="url", iiurlwidth=640, format="json"))
    request = urllib.request.Request("https://commons.wikimedia.org/w/api.php?"+params,
                                     headers={"User-Agent":"EcoSortCourseworkReproduction/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        page = next(iter(json.load(response)["query"]["pages"].values()))
    info = page["imageinfo"][0]
    url = info.get("thumburl", info["url"])
    assert urlparse(url).scheme == "https" and urlparse(url).netloc == "upload.wikimedia.org"
    with urllib.request.urlopen(url, timeout=40) as response:
        data = response.read()
    assert hashlib.sha256(data).hexdigest() == item["sha256"], "Source changed; do not replace a test image."
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=BASE/"evidence/photos")
    parser.add_argument("--restore", action="store_true", help="Restore missing source photos; stop on changed bytes or HTTP errors.")
    args = parser.parse_args()
    rule_checks()
    evidence = BASE/"evidence"
    manifest = json.loads((evidence/"original_manifest.json").read_text())
    original = json.loads((evidence/"original_results.json").read_text())
    assert len(manifest["images"]) == len(original["predictions"]) == 60
    expected = {r["id"]:r for r in original["predictions"]}
    paths = []
    for item in manifest["images"]:
        path = args.images/item["file"]
        assert path.parent.resolve() == args.images.resolve()
        if not path.exists() and args.restore:
            restore(item, path)
        assert path.is_file(), f"Missing {path.name}; supply --images or --restore."
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        paths.append(path)
    engine = EcoSort()
    for bad in (None, "not an image"):
        try:
            engine.classify(bad)
            raise AssertionError("Invalid input was accepted.")
        except gr.Error:
            pass
    rows = []
    for item, path in zip(manifest["images"], paths):
        with Image.open(path) as image:
            scores, decision, guidance = engine.classify(image)
        row = expected[item["id"]]
        assert row["predicted_route"] in decision and row["predicted_object"] in decision
        assert DISPLAY[row["prediction"]] in decision
        assert decision.startswith("Unable to identify reliably") == row["uncertain"]
        assert max(scores, key=scores.get) == DISPLAY[row["prediction"]]
        assert np.isfinite(list(scores.values())).all() and np.isclose(sum(scores.values()), 1)
        assert "User-confirmed" not in guidance
        rows.append(dict(id=item["id"], decision=decision, matches_original=True))
    with Image.open(paths[0]) as image:
        assert "red lid" in engine.classify(image, "Soft plastic bag or wrapper")[1]
        try:
            engine.classify(image, "invalid item")
            raise AssertionError("Invalid confirmation was accepted.")
        except gr.Error:
            pass
    code = ["app.py", "classifier.py", "disposal_rules.py", "evaluate.py"]
    record = dict(verified_at=datetime.now(ZoneInfo("Australia/Sydney")).isoformat(),
                  kind="Behavior-equivalence regression on the original, already-observed acceptance panel; not fresh validation.",
                  parity_count=len(rows), all_decisions_match=True, original_groups=original["groups"],
                  decisions=rows, rule_checks_passed=True, invalid_input_checks_passed=True,
                  code_sha256={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in code},
                  evidence_sha256={name:hashlib.sha256((evidence/name).read_bytes()).hexdigest()
                                   for name in ("original_manifest.json", "original_results.json")})
    target = evidence/"refactor_verified.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(target)
    print("Rule and input checks passed. All 60 category/object/review/route decisions match the original, including failures.")
    print(json.dumps(original["groups"], indent=2))


if __name__ == "__main__":
    main()
