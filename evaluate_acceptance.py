"""Evaluate every frozen acceptance image once; preserve failures and provenance."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
from object_model import ObjectPredictor, REVISION, SHA256

BASE = Path(__file__).resolve().parent
OUT = BASE/"acceptance_web"


def main():
    target = OUT/"results.json"
    assert not target.exists(), "Never overwrite acceptance evidence."
    manifest_path = OUT/"test_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["model_files_sha256"].items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest() == digest, "Model changed after panel freeze."
    model = ObjectPredictor()
    rows = []
    for item in manifest["images"]:
        path = Path(item["path"])
        if not path.is_file():
            path = OUT/path.name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        began = time.perf_counter()
        with Image.open(path) as image:
            result = model.identify(image, food_service=False)
        category = result["label"] == item["expected_label"]
        route = result["route"] == item["expected_route"]
        row = dict(panel=item["panel"], id=item["id"], expected_label=item["expected_label"],
                   expected_object=item["expected_object"], expected_route=item["expected_route"],
                   prediction=result["label"], predicted_object=result["object"], predicted_route=result["route"],
                   uncertain=result["uncertain"], category_correct=category, route_correct=route,
                   joint_correct=bool(category and route and not result["uncertain"]),
                   seconds=time.perf_counter()-began, source_url=item["source_url"], sha256=item["sha256"])
        rows.append(row)
        print(json.dumps(row), flush=True)
    groups = []
    for panel in (1,2,3):
        subset = [r for r in rows if r["panel"] == panel]
        assert len(subset) == 20
        counts = {key:sum(r[key] for r in subset) for key in ("category_correct", "route_correct", "joint_correct")}
        groups.append(dict(panel=panel, total=20, **counts, passed=counts["joint_correct"] >= 17))
    results = dict(evaluated_at=datetime.now(timezone.utc).isoformat(), groups=groups,
                   passed=all(r["passed"] for r in groups), predictions=rows,
                   manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   model_revision=REVISION, model_weight_sha256=SHA256,
                   model_files_sha256=manifest["model_files_sha256"],
                   warning="Conditional object-level route matching, not confirmation of contents or real-world population accuracy. No ground-truth item confirmation was supplied to inference.")
    target.write_text(json.dumps(results, indent=2))
    with (OUT/"results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("SUMMARY", json.dumps(groups), "PASSED", results["passed"], flush=True)


if __name__ == "__main__":
    main()
