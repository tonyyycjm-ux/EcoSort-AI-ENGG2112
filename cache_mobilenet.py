"""Cache unchanged MobileNet probabilities, preserving manifest row order."""
import hashlib
import json
import numpy as np
import pandas as pd
from train_full import BASE, dataset, tf

if __name__ == "__main__":
    out = BASE / "trained_semantic"
    out.mkdir(exist_ok=True)
    manifest_path = BASE / "data/expanded/split_manifest.csv"
    frame = pd.read_csv(manifest_path)
    model = tf.keras.models.load_model(BASE / "trained_expanded/ecosort_model.keras", compile=False)
    for split in ("validation", "test"):
        rows = frame[frame.split == split].sort_values(["label", "path"]).reset_index(drop=True)
        values = model.predict(dataset(rows), verbose=0)
        assert values.shape == (len(rows), 17) and np.isfinite(values).all()
        np.save(out / f"mobilenet_{split}.npy", values)
        print("Cached unchanged MobileNet:", split, len(rows), flush=True)
    (out / "mobilenet_cache.json").write_text(json.dumps({"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "model_sha256":hashlib.sha256((BASE / "trained_expanded/ecosort_model.keras").read_bytes()).hexdigest()}, indent=2))
