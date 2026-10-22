"""Freeze visually reviewed labels and routes before any acceptance prediction."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
from prepare_expanded import audit

BASE = Path(__file__).resolve().parent
OUT = BASE/"acceptance_web"
CHOICES = [
    ["0:0", "0:1", "0:2", "0:4"], ["1:0", "1:1", "1:2", "1:3"],
    ["2:0", "2:2", "2:4"], ["3:0", "3:2", "3:3"],
    ["21:0", "21:1", "21:2"], ["5:5", "29:0", "29:1", "29:3"],
    ["6:0", "6:2", "6:4", "6:5"], ["7:1", "7:3", "7:4"],
    ["8:3", "8:4", "20:1", "20:2"], ["9:3", "9:4", "9:5"],
    ["10:2", "10:4", "10:5"], ["11:0", "11:1", "11:2"],
    ["12:3", "31:1", "35:0", "37:0"], ["27:0", "32:1", "36:0"],
    ["14:2", "14:3", "14:5"], ["15:0", "15:1", "15:2", "15:4", "15:5"],
    ["16:0", "16:1", "16:4", "16:5"], ["17:1", "17:2", "17:3", "17:4", "17:5"],
    ["18:0", "18:3", "18:4"], ["25:0", "25:1", "25:3", "25:4"],
]
OBJECTS = ["Plastic drink bottle", "Rigid plastic food packaging", "Glass bottle", "Glass jar",
           "Aluminium beverage can", "Metal food tin", "Cardboard shipping box", "Newspaper",
           "Household battery", "Computer keyboard", "Fabric shirt", "Porcelain mug",
           "Food scraps / fruit peel", "Fallen leaves", "Wooden spoon", "Soft plastic shopping bag",
           "Paper kitchen towel", "Compact fluorescent lamp", "Medicine tablets in packaging", "Paint in cans"]
ROUTES = ["Recyclable - yellow lid bin"]*8 + [
    "Problem waste - battery drop-off", "Special recycling - e-waste collection",
    "Reuse or special textile recycling", "Non-recyclable in yellow lid - confirm disposal",
    "Food organics - confirm service; otherwise red lid", "Garden organics - green lid bin",
    "Wood item - check council disposal", "General waste - red lid bin", "General waste - red lid bin",
    "Problem waste - identify item and collection point", "Special disposal - community pharmacy",
    "Problem waste - identify item and collection point"]


def main():
    target = OUT/"test_manifest.json"
    assert not target.exists(), "Never replace a frozen acceptance panel."
    candidates = {r["id"]: r for r in json.loads((OUT/"candidates.json").read_text())}
    pool = pd.read_csv(BASE/"data/expanded/raw_audit.csv", dtype={"phash": "string"})
    pool = pool[pool.status == "valid"]
    sha, pixels = set(pool.sha256), set(pool.pixel_hash)
    hashes = [int(value) for value in pool.phash]
    for folder in ("external_test", "final_test", "fresh_test"):
        for row in json.loads((BASE/folder/"test_manifest.json").read_text())["images"]:
            checked = audit(row)
            if checked["status"] == "valid":
                sha.add(checked["sha256"])
                pixels.add(checked["pixel_hash"])
                hashes.append(int(checked["phash"]))
    chosen, rejected, missing = [], [], []
    for slot, identities in enumerate(CHOICES):
        selected = []
        for identity in identities:
            if identity not in candidates:
                continue
            row = candidates[identity]
            checked = audit(row)
            assert checked["status"] == "valid"
            distance = min((int(checked["phash"]) ^ h).bit_count() for h in hashes)
            if checked["sha256"] in sha or checked["pixel_hash"] in pixels or distance <= 2:
                rejected.append(dict(id=identity, reason="Project pool/previous panel/selected-photo duplicate heuristic", distance=distance))
                continue
            route = "General waste - red lid bin" if identity == "1:0" else ROUTES[slot]
            selected.append(dict(**row, panel=len(selected)+1, acceptance_slot=slot, expected_object=OBJECTS[slot],
                                 expected_route=route, minimum_phash_distance=distance,
                                 disposal_conditions="Discarded item; empty/dry accepted packaging, remove food; service access and hazards must be confirmed. No user-confirmed item is supplied to the model."))
            sha.add(checked["sha256"])
            pixels.add(checked["pixel_hash"])
            hashes.append(int(checked["phash"]))
            if len(selected) == 3:
                break
        chosen.extend(selected)
        if len(selected) != 3:
            missing.append(slot)
    review = dict(selected=chosen, rejected=rejected, missing_slots=missing)
    (OUT/"selection_review.json").write_text(json.dumps(review, indent=2))
    print("Selected", len(chosen), "Missing slots", missing, "Rejected", rejected, flush=True)
    assert len(chosen) == 60 and not missing, "Complete visual review and duplicate replacements before freezing."
    assert len({r["sha256"] for r in chosen}) == 60
    assert all(sum(r["panel"] == panel for r in chosen) == 20 for panel in (1,2,3))
    model_files = ["object_model.py", "semantic_model.py", "sydney_rules.py"]
    manifest = dict(frozen_at=datetime.now(timezone.utc).isoformat(), images=chosen,
                    protocol="Three stratified convenience panels; 20 whole photos each, same predefined object slots. Visually labelled before prediction. Not a random population sample. No trial stopping on successful pictures.",
                    pass_rule="Each panel needs at least 17/20 category AND conditional disposal-route matches; uncertain/manual-review outputs count as failures. Separate category-only accuracy is secondary.",
                    limitations="No claim of unseen foundation-model pretraining. Duplicate checks are heuristic. City of Sydney routes are conditional, not an image-based guarantee of contents or safety. Mixed-items class is not covered.",
                    input_settings=dict(food_service=False, item_type="Not confirmed"),
                    model_files_sha256={name: hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in model_files},
                    rejected_before_prediction=rejected)
    target.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
