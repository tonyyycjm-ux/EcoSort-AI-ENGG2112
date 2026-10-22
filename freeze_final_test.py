"""Freeze whole-photo choices after visual review, before model predictions."""
from datetime import datetime,timezone
import json
from pathlib import Path
import pandas as pd
from prepare_expanded import audit

BASE=Path(__file__).resolve().parent
OUT=BASE/"final_test"
CHOICES={
    "plastic":["0:1","0:2","0:3","0:0"],
    "glass":["1:0","1:4","13:0"],
    "ceramics":["2:0","2:1","2:2","2:5"],
    "metal":["3:1","3:2","3:0"],
    "cardboard":["4:1","4:2","4:3","4:4"],
    "paper":["5:1","5:2","5:3","5:5"],
    "battery":["6:0","6:1","6:2","6:3","6:4","6:5"],
    "ewaste":["7:0","7:2","7:3","7:5"],
    "food_organics":["8:0","8:2","8:4","15:3"],
    "textile_trash":["9:0","12:2","12:5","12:4"],
}

if __name__=="__main__":
    target=OUT/"test_manifest.json"
    assert not target.exists(),"Panel already frozen."
    candidates={r["id"]:r for r in json.loads((OUT/"candidates.json").read_text())}
    pool=pd.read_csv(BASE/"data/expanded/raw_audit.csv",dtype={"phash":"string"})
    pool=pool[pool.status=="valid"]
    sha,pixels=set(pool.sha256),set(pool.pixel_hash)
    hashes=[int(value) for value in pool.phash]
    for row in json.loads((BASE/"external_test/test_manifest.json").read_text())["images"]:
        checked=audit(row)
        sha.add(checked["sha256"])
        pixels.add(checked["pixel_hash"])
        hashes.append(int(checked["phash"]))
    selected,rejected=[],[]
    for label,ids in CHOICES.items():
        group=[]
        for identity in ids:
            row=candidates[identity]
            assert row["expected_label"]==label
            checked=audit(row)
            assert checked["status"]=="valid"
            distance=min((int(checked["phash"])^h).bit_count() for h in hashes)
            if checked["sha256"] in sha or checked["pixel_hash"] in pixels or distance<=2:
                rejected.append({"id":identity,"reason":"Exact/pixel/perceptual match against project pool or previous test; before predictions","distance":distance})
                continue
            group.append({**row,"minimum_phash_distance":distance})
            hashes.append(int(checked["phash"]))
            sha.add(checked["sha256"])
            pixels.add(checked["pixel_hash"])
            if len(group)==3:
                break
        selected.extend(group)
        print(label,len(group),[r["id"] for r in group],flush=True)
    (OUT/"selection_review.json").write_text(json.dumps({"selected":selected,"rejected":rejected},indent=2))
    assert len(selected)==30,"Need three valid distinct photos per class before freezing."
    target.write_text(json.dumps({"frozen_at":datetime.now(timezone.utc).isoformat(),"coverage":"Thirty unmodified whole photos, three per each of ten common visual categories; not all seventeen categories.","limitations":"Convenience sample, not Sydney-local or a guarantee for future photos. Project training duplicates and previous ten test photos excluded with heuristic checks; foundation-model pretraining membership is unknown. No model predictions consulted during selection.","selection":"Visually reviewed real photos; reject drawings, unrelated scenes, ambiguous materials and repeated views of the same object. Priority order fixed before any model prediction; fallback only for duplicate checks.","rejected_before_prediction":rejected,"images":selected},indent=2))
    print("Frozen 30 whole photos.")
