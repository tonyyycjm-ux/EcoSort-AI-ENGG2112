"""Local object-level SigLIP2 candidate, separate from benchmark classifiers."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import ImageOps
from transformers import AutoModel, AutoProcessor
from semantic_model import ITEMS
from sydney_rules import predicted_item_guidance

BASE = Path(__file__).resolve().parent
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
OBJECT_ITEMS = {label: list(items) for label, items in ITEMS.items()}
OBJECT_ITEMS["plastic"] += ["plastic toy", "plastic jigsaw puzzle pieces", "black plastic food container"]
OBJECT_ITEMS["trash"] += ["paper kitchen towel", "roll of paper kitchen towel", "paper napkin"]
OBJECT_ITEMS["vegetation"] += ["pile of fallen leaves", "heap of grass clippings"]
OBJECT_ITEMS["food_organics"] += ["banana peel", "orange peel", "apple core", "food waste"]
OBJECT_ITEMS["metal"] += ["metal food can", "empty tin can"]


class ObjectPredictor:
    def __init__(self):
        folder = BASE/"models/siglip2"
        with (folder/"model.safetensors").open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == SHA256
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch.set_num_threads(8)
        self.processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        self.model = AutoModel.from_pretrained(folder, local_files_only=True).to(self.device).eval()
        self.classes = sorted(ITEMS)
        self.items = [(label, item) for label, items in OBJECT_ITEMS.items() for item in items]
        tokens = self.processor(text=[f"This is a photo of {item}." for _, item in self.items],
                                padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            text = self.model.get_text_features(**tokens.to(self.device)).pooler_output
            self.text = text / text.norm(dim=-1, keepdim=True)

    def predict(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            features = self.model.get_image_features(**inputs).pooler_output
            features /= features.norm(dim=-1, keepdim=True)
            logits = (features @ self.text.T * self.model.logit_scale.exp() + self.model.logit_bias)[0]
        values = logits.float().cpu().numpy()
        scores = np.asarray([max(values[i] for i, pair in enumerate(self.items) if pair[0] == label)
                             for label in self.classes])
        assert np.isfinite(scores).all() and scores.shape == (17,)
        best = int(values.argmax())
        return self.items[best], scores

    def identify(self, image, food_service=False):
        (label, name), logits = self.predict(image)
        ordered = np.sort(logits)
        uncertain = bool(ordered[-1]-ordered[-2] < .5 or ordered[-1] < -10)
        route, guidance = predicted_item_guidance(label, name, food_service, uncertain)
        relative = np.exp(logits-logits.max())
        relative /= relative.sum()
        return dict(label=label, object=name, uncertain=uncertain, route=route, guidance=guidance,
                    scores=dict(zip(self.classes, map(float, relative))))


if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    snapshot_download("google/siglip2-base-patch16-224", revision=REVISION,
                      local_dir=BASE/"models/siglip2", allow_patterns=["*.json", "tokenizer.model", "model.safetensors"])
    ObjectPredictor()
    print("Pinned object model loaded and weight checksum verified.")
