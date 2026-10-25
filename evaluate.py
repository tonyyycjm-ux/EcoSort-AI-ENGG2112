"""Re-evaluate the project-trained model on the fixed validation and test splits."""
import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from app import BASE, EcoSort
from train import dataset, load_manifest, LABELS
from disposal_rules import disposal_guidance


def main():
    manifest_path = BASE / "data/expanded/split_manifest.csv"
    manifest = load_manifest()
    audit = json.loads((BASE / "data/expanded/dataset_verification.json").read_text())
    assert len(manifest) == audit["retained_images"] == 147303
    assert manifest.sha256.is_unique and manifest.groupby("group_id").split.nunique().max() == 1
    assert manifest.groupby("split").size().to_dict() == audit["split_counts"]
    assert all(Path(p).is_file() for p in manifest.path)
    engine = EcoSort()
    assert engine.classifier.classes == LABELS
    for bad in (None, "not an image"):
        try:
            engine.classify(bad)
            raise AssertionError("Invalid image accepted.")
        except gr.Error:
            pass
    assert "red" in disposal_guidance("plastic", "Soft plastic bag or wrapper")[0]
    assert "drop-off" in disposal_guidance("battery")[0]
    assert "Potentially" in disposal_guidance("plastic")[0]
    metrics = {}
    for split in ("validation", "test"):
        frame = manifest[manifest.split == split].sort_values(["label", "path"]).reset_index(drop=True)
        truth = frame.label.map(dict(zip(LABELS, range(len(LABELS))))).to_numpy()
        probabilities = engine.classifier.model.predict(dataset(frame), verbose=1)
        assert probabilities.shape == (len(frame), len(LABELS)) and np.isfinite(probabilities).all()
        predicted = probabilities.argmax(axis=1)
        metrics[split] = dict(images=len(frame), accuracy=float(accuracy_score(truth, predicted)),
                              macro_f1=float(f1_score(truth, predicted, average="macro", zero_division=0)))
        print(split, metrics[split], flush=True)
        # Confirm deployment preprocessing agrees with batch evaluation.
        for index in (0, len(frame)//2, len(frame)-1):
            with Image.open(frame.path.iloc[index]) as image:
                scores, decision, guidance = engine.classify(image)
                single = engine.classifier.probabilities(image)
                assert int(single.argmax()) == int(predicted[index])
                assert np.isclose(sum(scores.values()), 1)
                assert "SigLIP" not in decision
                assert "yellow lid bin" not in decision or "Potentially" in decision
                confirmed = engine.classify(image, "Soft plastic bag or wrapper")
                assert "red lid" in confirmed[1]
    code = ["app.py", "classifier.py", "disposal_rules.py", "evaluate.py", "train.py"]
    artifacts = ["ecosort_model.keras", "app_settings.json", "training_completed.json"]
    record = dict(model="MobileNetV3Large", initialization="ImageNet; subsequently trained on project images",
                  training_images=audit["split_counts"]["train"], retained_images=len(manifest),
                  verified_at=datetime.now(ZoneInfo("Australia/Sydney")).isoformat(),
                  metrics=metrics, rules_and_inputs_passed=True,
                  manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  code_sha256={n:hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in code},
                  artifact_sha256={n:hashlib.sha256((BASE/"trained_expanded"/n).read_bytes()).hexdigest() for n in artifacts},
                  limitations="Previously observed dataset splits, not fresh independent validation. No real-world accuracy guarantee.")
    target = BASE / "evidence/trained_model_verified.json"
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(target)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
