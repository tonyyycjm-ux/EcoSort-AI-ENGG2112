"""Development-only object prompts; no fresh acceptance-panel tuning."""
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageOps

from semantic_model import ITEMS, image_features, load_encoder

BASE = Path(__file__).resolve().parent


def main():
    model, transform, device = load_encoder()
    prompts = [(label, item) for label, items in ITEMS.items() for item in items]
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer([f"a photo of a {item}." for _, item in prompts]).to(device)
    with torch.inference_mode():
        text = model.encode_text(tokens).float()
        text /= text.norm(dim=-1, keepdim=True)
    text = text.cpu().numpy()
    panels = {"local_development": [dict(path=str(Path.home()/"Downloads"/f"IMG_{n}.JPG"), expected_label=y)
              for n, y in zip(range(9413, 9419), ["ceramics", "cardboard", "cardboard", "plastic", "plastic", "trash"])]}
    for name in ("external_test", "final_test"):
        panels[name] = json.loads((BASE/name/"test_manifest.json").read_text())["images"]
    results = []
    for panel, items in panels.items():
        correct = 0
        for item in items:
            with Image.open(item["path"]) as image:
                features = image_features(model, transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0), device)
            similarities = (features @ text.T)[0]
            order = np.argsort(similarities)[::-1][:3]
            label = prompts[order[0]][0]
            correct += label == item["expected_label"]
            row = dict(panel=panel, file=Path(item["path"]).name, expected=item["expected_label"],
                       prediction=label, candidates=[prompts[i] for i in order])
            results.append(row)
            print(json.dumps(row), flush=True)
        print(panel, correct, len(items), flush=True)
    output = BASE/"object_prompt_development.json"
    output.write_text(json.dumps(results, indent=2))
    assert len(results) == sum(map(len, panels.values()))


if __name__ == "__main__":
    main()
