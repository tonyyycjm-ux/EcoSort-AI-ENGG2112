"""Freeze visually reviewed external photos before any model evaluation."""
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from prepare_expanded import audit

BASE = Path(__file__).resolve().parent
OUT = BASE / "external_test"

if __name__ == "__main__":
    target = OUT / "test_manifest.json"
    assert not target.exists(), "The evaluation panel is already frozen."
    # The unobstructed keyboard candidate matched the pool at perceptual distance 2.
    choices = [2, 0, 0, 3, 1, 0, 0, 1, 0, 2]
    candidates = json.loads((OUT / "candidates.json").read_text())
    pool = pd.read_csv(BASE / "data/expanded/raw_audit.csv", dtype={"phash":"string"})
    valid = pool[pool.status == "valid"]
    hashes = [int(value) for value in valid.phash]
    selected = []
    for index, rank in enumerate(choices):
        entry = next(row for row in candidates if row["test_index"] == index and row["candidate_rank"] == rank)
        checked = audit(entry)
        assert checked["status"] == "valid", entry["title"]
        exact = bool((valid.sha256 == checked["sha256"]).any() or (valid.pixel_hash == checked["pixel_hash"]).any())
        distance = min((int(checked["phash"]) ^ value).bit_count() for value in hashes)
        assert not exact and distance > 2, f"Duplicate candidate: {entry['title']}, distance={distance}"
        selected.append({**entry, "minimum_pool_phash_distance":distance, "exact_duplicate":exact})
    assert len(selected) == 10
    target.write_text(json.dumps({"frozen_at":datetime.now(timezone.utc).isoformat(), "selection":"Visual clarity and identifiable material; no model predictions used.", "duplicate_check":"Against the full audited raw pool: SHA256, resized pixels, canonical perceptual hash distance > 2. Heuristic, not a guarantee of independence.", "images":selected}, indent=2))
    print("Frozen 10 externally sourced images before model evaluation.")
