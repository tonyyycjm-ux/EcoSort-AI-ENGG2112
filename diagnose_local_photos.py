"""Diagnose user photos without fitting models or changing deployment."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.special import softmax

from ecosort_runtime import BASE, EcoSort


def main():
    engine = EcoSort()
    # Manually selected target regions are diagnostic only, not automatic detection.
    boxes = [(0.05, .12, 1, .95), (0, 0, 1, 1), (0, .05, 1, .91),
             (.08, .27, .93, .98), (.18, 0, .69, .92), (.05, .16, .87, .95)]
    rows = []
    for number, box in zip(range(9413, 9419), boxes):
        path = Path.home() / "Downloads" / f"IMG_{number}.JPG"
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        for mode in ("whole", "target_crop"):
            target = image if mode == "whole" else image.crop(tuple(round(v*d) for v, d in zip(box, (*image.size, *image.size))))
            label, confidence, review, scores = engine.predict(target)
            pixels = np.asarray(target.resize((224,224), Image.Resampling.BILINEAR), dtype=np.float32)[None]
            raw = engine.model(pixels, training=False).numpy()[0]
            linear, zero = engine.semantic.logits(target)
            def top(values):
                return [(engine.classes[int(i)], round(float(values[i]), 4)) for i in np.argsort(values)[-3:][::-1]]
            row = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mode=mode,
                       box=box if mode != "whole" else None, prediction=label, confidence=confidence,
                       review=review, ensemble=top(np.array(list(scores.values()))),
                       mobilenet=top(softmax(np.log(np.clip(raw, 1e-8, 1))/engine.mobile_temperature)),
                       trained_clip=top(softmax(linear/engine.settings["linear_temperature"])),
                       zero_shot_diagnostic=top(softmax(zero/engine.settings["zero_temperature"])))
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = BASE / "local_photo_diagnostic"
    out.mkdir(exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(rows, indent=2))
    assert len(rows) == 12 and all(0 <= r["confidence"] <= 1 for r in rows)


if __name__ == "__main__":
    main()
