"""Project-trained MobileNetV3; no image-text or zero-shot fallback."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from disposal_rules import disposal_guidance

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "trained_expanded"


class TrainedPredictor:
    def __init__(self):
        completed = json.loads((RESULTS / "training_completed.json").read_text())
        assert completed["status"] == "completed"
        self.settings = json.loads((RESULTS / "app_settings.json").read_text())
        self.classes = self.settings["classes"]
        assert self.settings["training_images"] == completed["split_counts"]["train"]
        assert completed["verified_image_count"] == sum(completed["split_counts"].values())
        self.model = tf.keras.models.load_model(RESULTS / "ecosort_model.keras", compile=False)
        assert self.model.output_shape[-1] == len(self.classes) == 17

    def probabilities(self, image):
        rgb = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        raw = self.model(np.asarray(rgb, dtype=np.float32)[None, ...], training=False).numpy()[0]
        logits = np.log(np.clip(raw, 1e-8, 1)) / self.settings["temperature"]
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        assert scores.shape == (len(self.classes),) and np.isfinite(scores).all()
        return scores

    def identify(self, image, food_service=False):
        scores = self.probabilities(image)
        order = np.argsort(scores)
        label = self.classes[int(order[-1])]
        uncertain = bool(scores[order[-1]] < self.settings["review_threshold"]
                         or scores[order[-1]] - scores[order[-2]] < self.settings["minimum_top_two_margin"]
                         or label in self.settings["manual_review_classes"])
        # Material-only training cannot establish bottle, bag or container identity.
        route, guidance = disposal_guidance(label, food_service=food_service, uncertain=uncertain)
        return dict(label=label, object=label.replace("_", " "), uncertain=uncertain,
                    route=route, guidance=guidance, scores=dict(zip(self.classes, map(float, scores))))


if __name__ == "__main__":
    predictor = TrainedPredictor()
    print("Loaded project-trained MobileNetV3Large:", predictor.settings["training_images"], "training images.")
