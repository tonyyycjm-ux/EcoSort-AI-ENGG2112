"""Fit a frozen semantic encoder on all training rows; select on validation only."""
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from semantic_model import BASE, REVISION, SHA256, PREPROCESS_VERSION, Images, combine, image_features, load_encoder, text_features

OUT = BASE / "trained_semantic"
OUT.mkdir(exist_ok=True)

def extract(model, transform, device, frame, split, signature):
    path = OUT / f"features_{split}.npy"
    marker = OUT / f"features_{split}.json"
    saved = json.loads(marker.read_text()) if marker.exists() else {}
    assert not saved or saved["manifest_sha256"] == signature, "The cached feature order changed."
    assert not saved or saved.get("preprocess_version") == PREPROCESS_VERSION, "The preprocessing changed; rebuild these features."
    if saved.get("completed"):
        values = np.load(path, mmap_mode="r")
        assert values.shape == (len(frame), 512) and np.isfinite(values).all()
        return values
    values = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float32, shape=(len(frame), 512))
    start = saved.get("rows", 0)
    loader = DataLoader(Images(frame.path.iloc[start:].to_list(), transform), batch_size=64, shuffle=False, num_workers=2, persistent_workers=True)
    began = time.perf_counter()
    position = start
    for batch_index, batch in enumerate(loader):
        features = image_features(model, batch, device)
        assert np.isfinite(features).all()
        values[position:position + len(features)] = features
        position += len(features)
        if batch_index % 25 == 0 or position == len(frame):
            values.flush()
            marker.write_text(json.dumps({"manifest_sha256":signature, "preprocess_version":PREPROCESS_VERSION, "rows":position, "completed":position == len(frame)}))
            rate = (position - start) / (time.perf_counter() - began)
            print(f"FEATURES {split}: {position}/{len(frame)}, {rate:.1f} images/s, remaining {(len(frame)-position)/max(rate,1)/60:.1f} min", flush=True)
    assert position == len(frame)
    return values

def temperature(logits, truth):
    def loss(value):
        p = softmax(logits / value, axis=1)
        return float(-np.log(np.clip(p[np.arange(len(truth)), truth], 1e-9, 1)).mean())
    return float(minimize_scalar(loss, bounds=(0.25, 5), method="bounded").x)

def metrics(probs, truth):
    return {"accuracy":float(accuracy_score(truth, probs.argmax(1))), "macro_f1":float(f1_score(truth, probs.argmax(1), average="macro", zero_division=0))}

def main():
    started = time.perf_counter()
    source = BASE / "data/expanded/split_manifest.csv"
    signature = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = pd.read_csv(source)
    assert manifest.groupby("group_id").split.nunique().max() == 1 and manifest.sha256.is_unique
    classes = json.loads((BASE / "data/expanded/class_names.json").read_text())
    assert set(classes) == set(manifest.label)
    frames = {split:manifest[manifest.split == split].sort_values(["label", "path"]).reset_index(drop=True) for split in ("train", "validation", "test")}
    truth = {split:frame.label.map({label:i for i,label in enumerate(classes)}).to_numpy() for split,frame in frames.items()}
    model, transform, device = load_encoder()
    print("Frozen semantic encoder:", device, "All training rows:", len(frames["train"]), flush=True)
    features = {split:extract(model, transform, device, frame, split, signature) for split,frame in frames.items()}
    texts = text_features(model, classes, device)
    scale = float(model.logit_scale.exp().detach().cpu())
    np.save(OUT / "text_features.npy", texts)
    counts = np.bincount(truth["train"], minlength=len(classes))
    weights = np.minimum(np.sqrt(len(frames["train"]) / (len(classes) * counts)), 3) ** 0.5
    trials = []
    best = None
    for strength in (1.0, 10.0, 100.0):
        classifier = LogisticRegression(C=strength, max_iter=300, class_weight=dict(enumerate(weights)), random_state=42)
        classifier.fit(features["train"], truth["train"])
        assert np.array_equal(classifier.classes_, np.arange(len(classes)))
        p = classifier.predict_proba(features["validation"])
        measured = metrics(p, truth["validation"])
        converged = bool(classifier.n_iter_.max() < classifier.max_iter)
        trials.append({"C":strength, "converged":converged, **measured})
        print("VALIDATION linear head:", trials[-1], flush=True)
        if converged and (best is None or (measured["accuracy"], measured["macro_f1"]) > best[0]):
            best = ((measured["accuracy"], measured["macro_f1"]), classifier)
    assert best is not None, "No converged classifier is available."
    classifier = best[1]
    np.savez(OUT / "classifier.npz", coef=classifier.coef_, intercept=classifier.intercept_, text=texts)
    linear_val = classifier.decision_function(features["validation"])
    zero_val = features["validation"] @ texts.T * scale
    linear_temp = temperature(linear_val, truth["validation"])
    zero_temp = temperature(zero_val, truth["validation"])
    old = json.loads((BASE / "trained_expanded/app_settings.json").read_text())
    cache = json.loads((OUT / "mobilenet_cache.json").read_text())
    assert cache["manifest_sha256"] == signature
    assert cache["model_sha256"] == hashlib.sha256((BASE / "trained_expanded/ecosort_model.keras").read_bytes()).hexdigest(), "MobileNet cache/model mismatch."
    mobile_val = softmax(np.log(np.clip(np.load(OUT / "mobilenet_validation.npy"), 1e-8, 1)) / old["temperature"], axis=1)
    settings = {"classes":classes, "linear_temperature":linear_temp, "zero_temperature":zero_temp, "zero_logit_scale":scale}
    candidates = []
    for mobile_weight in (0, 0.25, 0.5, 0.75, 1):
        for zero_weight in (0, 0.1, 0.25, 0.5, 0.75, 1):
            if mobile_weight + zero_weight > 1:
                continue
            trial = {**settings, "weights":[mobile_weight, 1-mobile_weight-zero_weight, zero_weight]}
            p = combine(mobile_val, linear_val, zero_val, trial)
            candidates.append({**trial, **metrics(p, truth["validation"])})
    baseline = metrics(mobile_val, truth["validation"])
    eligible = [row for row in candidates if row["macro_f1"] >= baseline["macro_f1"]]
    selected = max(eligible, key=lambda row:(row["accuracy"],row["macro_f1"]))
    print("VALIDATION selected ensemble:", selected, flush=True)
    p = combine(mobile_val, linear_val, zero_val, selected)
    review_threshold = 0.95
    for threshold in np.arange(0.6, 0.96, 0.025):
        accepted = p.max(1) >= threshold
        if accepted.mean() >= 0.2 and (p.argmax(1)[accepted] == truth["validation"][accepted]).mean() >= 0.95:
            review_threshold = float(threshold)
            break
    selected.update({"review_threshold":review_threshold, "minimum_top_two_margin":0.2, "manual_review_classes":old["manual_review_classes"], "backbone":"MobileNetV3Large + frozen OpenCLIP ViT-B-32", "image_count":len(manifest), "training_images":len(frames["train"]), "temperature":1.0, "mobile_temperature":old["temperature"], "manifest_sha256":signature, "weights_revision":REVISION, "weights_sha256":SHA256, "selection":"Validation only; old external photos not used", "linear_trials":trials})
    selected["preprocess_version"] = PREPROCESS_VERSION
    selected["linear_C"] = float(classifier.C)
    selected["mobile_model_sha256"] = cache["model_sha256"]
    (OUT / "app_settings.json").write_text(json.dumps(selected, indent=2))
    mobile_test = softmax(np.log(np.clip(np.load(OUT / "mobilenet_test.npy"), 1e-8, 1)) / old["temperature"], axis=1)
    p = combine(mobile_test, classifier.decision_function(features["test"]), features["test"] @ texts.T * scale, selected)
    result = metrics(p, truth["test"])
    report = classification_report(truth["test"], p.argmax(1), target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(OUT / "classification_report.csv")
    np.savetxt(OUT / "confusion_matrix.csv", confusion_matrix(truth["test"], p.argmax(1)), delimiter=",", fmt="%d")
    np.save(OUT / "test_probabilities.npy", p)
    predicted = frames["test"][["dataset", "label", "path"]].copy()
    predicted["predicted"] = [classes[index] for index in p.argmax(1)]
    predicted.to_csv(OUT / "test_predictions.csv", index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k in ("weights", "accuracy", "macro_f1")} for row in candidates]).to_csv(OUT / "validation_candidates.csv", index=False)
    completion = {"status":"completed", "test_metrics":result, "test_accuracy_target_met":result["accuracy"] >= 0.8, "verified_image_count":len(manifest), "split_counts":{s:len(f) for s,f in frames.items()}, "seconds":time.perf_counter()-started, "note":"Public test was evaluated after validation selection. It was evaluated previously by the old model, so this is a benchmark comparison, not a newly collected real-world test."}
    (OUT / "training_completed.json").write_text(json.dumps(completion, indent=2))
    print(json.dumps(completion, indent=2), flush=True)

if __name__ == "__main__":
    main()
