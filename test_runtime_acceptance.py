"""Ensure the deployed UI callback reproduces all accepted and failed decisions."""
import json
from pathlib import Path
from PIL import Image
from ecosort_runtime import EcoSort, DISPLAY

BASE = Path(__file__).resolve().parent
manifest = json.loads((BASE/"acceptance_web/test_manifest.json").read_text())
record = json.loads((BASE/"acceptance_web/results.json").read_text())
expected = {r["id"]:r for r in record["predictions"]}
engine = EcoSort()
rows = []
for item in manifest["images"]:
    path = Path(item["path"])
    if not path.is_file():
        path = BASE/"acceptance_web"/path.name
    with Image.open(path) as image:
        scores, decision, guidance = engine.classify(image)
    row = expected[item["id"]]
    assert row["predicted_route"] in decision
    assert DISPLAY[row["prediction"]] in decision
    assert row["predicted_object"] in decision
    assert decision.startswith("Unable to identify reliably") == row["uncertain"]
    assert max(scores, key=scores.get) == DISPLAY[row["prediction"]]
    assert "User-confirmed" not in guidance
    rows.append(dict(id=item["id"], decision=decision, parity=True))
(BASE/"acceptance_web/runtime_parity.json").write_text(json.dumps(rows, indent=2))
assert len(rows) == 60
print("All 60 UI callback decisions match the frozen evaluation, including failures.")
