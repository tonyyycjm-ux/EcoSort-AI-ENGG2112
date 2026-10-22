"""First-pass new panel evaluation plus separately labelled old-panel regression."""
import json
import hashlib
import time
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from ecosort_runtime import BASE, EcoSort

def evaluate_panel(old, new, manifest, out, role):
    out.mkdir(exist_ok=True)
    assert not (out / "evaluation.json").exists(), "Preserve the first fresh evaluation."
    panel = json.loads(manifest.read_text())
    rows = []
    gallery = Image.new("RGB", (1400, 280*((len(panel["images"])+6)//7)), "white")
    draw = ImageDraw.Draw(gallery)
    for index, entry in enumerate(panel["images"]):
        path = Path(entry["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        with Image.open(path) as image:
            before = old.predict(image)[0]
            began = time.perf_counter()
            label, confidence, review, _ = new.predict(image)
            elapsed = time.perf_counter()-began
            x,y = index%7*200,index//7*280
            gallery.paste(ImageOps.contain(image.convert("RGB"),(195,180)),(x,y))
        correct = label == entry["expected_label"]
        draw.multiline_text((x+2,y+185),f"{entry['id']} Gold: {entry['expected_label']}\nNew: {label}\n{confidence:.1%} | Review: {review}\n{'CORRECT' if correct else 'INCORRECT'}",fill="green" if correct else "red",spacing=5)
        rows.append({**entry,"old_predicted":before,"predicted":label,"confidence":confidence,"manual_review":review,"correct":correct,"old_correct":before==entry["expected_label"],"seconds":elapsed})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "predictions.csv",index=False)
    report = {"images":len(frame),"correct":int(frame.correct.sum()),"accuracy":float(frame.correct.mean()),"old_correct":int(frame.old_correct.sum()),"old_accuracy":float(frame.old_correct.mean()),"manual_review_rate":float(frame.manual_review.mean()),"target_met":bool(frame.correct.mean() >= .8),"mean_seconds":float(frame.seconds.mean()),"scope":panel["coverage"],"limitations":panel["limitations"],"role":role,"selection":"Validation-selected model; role distinguishes the untouched final panel from previously inspected diagnostic photos."}
    (out / "evaluation.json").write_text(json.dumps(report,indent=2))
    frame.groupby("expected_label")[["correct","old_correct"]].agg(["sum","count","mean"]).to_csv(out / "per_class_results.csv")
    gallery.save(out / "results_gallery.jpg")
    return report

if __name__ == "__main__":
    old = EcoSort(BASE / "trained_expanded")
    new = EcoSort(BASE / "trained_semantic")
    report = evaluate_panel(old, new, BASE / "final_test/test_manifest.json", BASE / "final_test", "Independent final whole-photo panel; first model predictions")
    diagnostic = evaluate_panel(old, new, BASE / "fresh_test/test_manifest.json", BASE / "trained_semantic/taco_diagnostic", "Development regression: this panel exposed the center-crop issue; no longer independent")
    regression = []
    for entry in json.loads((BASE / "external_test/test_manifest.json").read_text())["images"]:
        with Image.open(entry["path"]) as image:
            label,confidence,review,_ = new.predict(image)
        regression.append({**entry,"predicted":label,"confidence":confidence,"manual_review":review,"correct":label==entry["expected_label"]})
    pd.DataFrame(regression).to_csv(BASE / "trained_semantic/old_10_regression.csv",index=False)
    (BASE / "trained_semantic/old_10_regression.json").write_text(json.dumps({"correct":sum(row["correct"] for row in regression),"images":10,"scope":"Previously seen panel: regression only, not an independent accuracy estimate."},indent=2))
    bottle = Path("PRIVATE_REFERENCE_NOT_INCLUDED")
    if bottle.exists():
        with Image.open(bottle) as image:
            label,confidence,review,scores = new.predict(image)
        (BASE / "trained_semantic/bottle_regression.json").write_text(json.dumps({"expected":"plastic","predicted":label,"confidence":confidence,"manual_review":review,"scores":scores},indent=2))
    print(json.dumps(report,indent=2),flush=True)
    print("TACO diagnostic regression:",json.dumps(diagnostic),flush=True)
