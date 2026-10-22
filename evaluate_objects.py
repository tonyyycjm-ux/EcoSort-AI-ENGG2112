"""Record object-model predictions on an explicitly supplied frozen manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
from object_model import ObjectPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), "Keep prior evaluation evidence."
    manifest = json.loads(args.manifest.read_text())
    model = ObjectPredictor()
    rows = []
    for item in manifest["images"]:
        path = Path(item["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        began = time.perf_counter()
        with Image.open(path) as image:
            (label, name), scores = model.predict(image)
        row = dict(id=item.get("id", path.name), expected=item["expected_label"], prediction=label,
                   object=name, correct=label == item["expected_label"], seconds=time.perf_counter()-began,
                   scores=dict(zip(model.classes, map(float, scores))))
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != "scores"}), flush=True)
    result = dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  correct=sum(r["correct"] for r in rows), total=len(rows), predictions=rows)
    args.output.write_text(json.dumps(result, indent=2))
    print("SUMMARY", result["correct"], result["total"], flush=True)


if __name__ == "__main__":
    main()
