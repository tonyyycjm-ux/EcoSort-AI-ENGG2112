"""Evaluate the frozen external panel once; never use it to select a model."""
import json
import time
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from ecosort_runtime import EcoSort

BASE = Path(__file__).resolve().parent
OUT = BASE / "external_test"

if __name__ == "__main__":
    assert not (OUT / "evaluation.json").exists(), "Preserve the first evaluation; do not overwrite it."
    panel = json.loads((OUT / "test_manifest.json").read_text())
    assert len(panel["images"]) == 10
    engine = EcoSort()
    rows = []
    gallery = Image.new("RGB", (1000, 1000), "white")
    draw = ImageDraw.Draw(gallery)
    for index, entry in enumerate(panel["images"]):
        with Image.open(entry["path"]) as image:
            started = time.perf_counter()
            label, confidence, review, scores = engine.predict(image)
            seconds = time.perf_counter() - started
            x, y = (index % 5) * 200, (index // 5) * 500
            gallery.paste(ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (195, 360)), (x, y))
        correct = label == entry["expected_label"]
        draw.multiline_text((x + 3, y + 365), f"Expected: {entry['expected_label']}\nPredicted: {label}\nScore: {confidence:.1%}\n{'CORRECT' if correct else 'INCORRECT'}\nReview: {review}", fill="green" if correct else "red", spacing=6)
        rows.append({**entry, "predicted":label, "confidence":confidence, "manual_review":review, "correct":correct, "seconds":seconds})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "predictions.csv", index=False)
    accepted = frame[~frame.manual_review]
    report = {"images":10, "correct":int(frame.correct.sum()), "top1_accuracy":float(frame.correct.mean()), "manual_review_rate":float(frame.manual_review.mean()), "accepted_count":len(accepted), "accepted_accuracy":float(accepted.correct.mean()) if len(accepted) else None, "mean_seconds":float(frame.seconds.mean()), "model_selection":"Validation only, before these predictions", "limitation":"Ten deliberately selected classes are a small convenience sample, not a population accuracy estimate."}
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2))
    gallery.save(OUT / "results_gallery.jpg")
    original = Path("PRIVATE_REFERENCE_NOT_INCLUDED")
    if original.exists():
        with Image.open(original) as image:
            label, confidence, review, scores = engine.predict(image)
        (OUT / "user_bottle_regression.json").write_text(json.dumps({"expected":"plastic", "predicted":label, "confidence":confidence, "manual_review":review, "scores":scores, "note":"Previously supplied user photo, separate from the ten-photo panel; not used in training."}, indent=2))
    print(json.dumps(report, indent=2))
