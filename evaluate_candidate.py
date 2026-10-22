"""Evaluate completed candidates on fixed diagnostics; never fit on these photos."""
import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from nonlinear_model import VisualPredictor

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    args = parser.parse_args()
    results = BASE/args.results
    assert results.parent == BASE
    completion = json.loads((results/"training_completed.json").read_text())
    assert completion["status"] == "completed"
    settings = json.loads((results/"selection.json").read_text())
    settings.setdefault("feature_source","trained_semantic")
    if "temperature" not in settings:
        selected = settings["selected"]
        def load(name):
            return np.load(results/(name.replace(":","_")+"_validation.npy"))
        p = selected["weight"]*load(selected["first"]) + (1-selected["weight"])*load(selected["second"])
        frame = pd.read_csv(BASE/"data/expanded/split_manifest.csv")
        frame = frame[frame.split=="validation"].sort_values(["label","path"])
        truth = frame.label.map(dict(zip(settings["classes"],range(len(settings["classes"]))))).to_numpy()
        logits = np.log(np.clip(p,1e-8,1))
        def nll(value):
            probability = softmax(logits/value,axis=1)
            return float(-np.log(np.clip(probability[np.arange(len(truth)),truth],1e-8,1)).mean())
        settings["temperature"] = float(minimize_scalar(nll,bounds=(.25,5),method="bounded").x)
    settings.update(review_threshold=.6,minimum_top_two_margin=.2,
                    manual_review_classes=["mixed_items","household_problem_waste","medicines","light_bulbs"])
    (results/"app_settings.json").write_text(json.dumps(settings,indent=2))
    engine = VisualPredictor(results)
    cached = np.load(BASE/settings["feature_source"]/"features_test.npy",mmap_mode="r")
    calculated = engine.from_features(np.asarray(cached[:8]).copy())
    saved = np.load(results/"test_probabilities.npy")[:8]
    if "temperature" not in completion:
        saved = softmax(np.log(np.clip(saved,1e-8,1))/settings["temperature"],axis=1)
    assert np.allclose(calculated,saved,atol=1e-4), "Training/inference probability mismatch."
    rows = []
    expected = ["ceramics","cardboard","cardboard","plastic","plastic","trash"]
    panels = {"local_six":[dict(path=str(Path.home()/"Downloads"/f"IMG_{number}.JPG"),expected_label=label)
                             for number,label in zip(range(9413,9419),expected)]}
    for name,manifest in (("web_thirty","final_test"),("taco_diagnostic","fresh_test"),("old_ten","external_test")):
        panels[name] = json.loads((BASE/manifest/"test_manifest.json").read_text())["images"]
    summaries = {}
    for panel,items in panels.items():
        count = 0
        for item in items:
            path = Path(item["path"])
            if "sha256" in item:
                assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            began = time.perf_counter()
            with Image.open(path) as image:
                p = engine.scores(image)
            label = settings["classes"][int(p.argmax())]
            order = np.sort(p)
            review = bool(p.max()<.6 or order[-1]-order[-2]<.2 or label in settings["manual_review_classes"])
            correct = label == item["expected_label"]
            count += correct
            row = dict(panel=panel,file=path.name,expected=item["expected_label"],prediction=label,
                       confidence=float(p.max()),review=review,correct=correct,seconds=time.perf_counter()-began)
            rows.append(row)
            print(json.dumps(row),flush=True)
        summaries[panel] = dict(correct=count,total=len(items),accuracy=count/len(items),
                               role="Previously observed development/regression panel; not new independent validation")
    (results/"external_diagnostics.json").write_text(json.dumps(summaries,indent=2))
    pd.DataFrame(rows).to_csv(results/"external_diagnostics.csv",index=False)
    print("DIAGNOSTICS",json.dumps(summaries),flush=True)


if __name__ == "__main__":
    main()
