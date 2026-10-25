"""Train MobileNetV3 on every training image in the audited expanded manifest."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["SSL_CERT_FILE"] = __import__("certifi").where()
import ssl
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
import tensorflow as tf

BASE = Path(__file__).resolve().parent
OUT = BASE / "trained_reproduction"
LABELS = json.loads((BASE / "data/expanded/class_names.json").read_text())
SIZE = (224, 224)
BATCH = 64

def load_manifest():
    frame = pd.read_csv(BASE / "data/expanded/split_manifest.csv")
    paths = [(BASE / path).resolve() for path in frame.path]
    assert all(path.is_relative_to(BASE / "data") for path in paths), "Dataset paths must stay inside this project."
    frame["path"] = [str(path) for path in paths]
    return frame

def load_image(path):
    if hasattr(path, "item"):
        path = path.item()
    with Image.open(path.decode()) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB").resize(SIZE, Image.Resampling.BILINEAR), dtype=np.float32)

def decode(path, label):
    pixels = tf.numpy_function(load_image, [path], tf.float32)
    pixels.set_shape([224, 224, 3])
    return pixels, label

def dataset(frame, training=False):
    values = frame.label.map({label: index for index, label in enumerate(LABELS)}).to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((frame.path.to_list(), values))
    if training:
        ds = ds.shuffle(len(frame), seed=42, reshuffle_each_iteration=True)
    return ds.map(decode, num_parallel_calls=4).batch(BATCH).prefetch(2)

def evaluate(name, probabilities, truth):
    predicted = probabilities.argmax(axis=1)
    report = classification_report(truth, predicted, labels=np.arange(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(OUT / (name + "_classification_report.csv"))
    np.savetxt(OUT / (name + "_confusion_matrix.csv"), confusion_matrix(truth, predicted, labels=np.arange(len(LABELS))), delimiter=",", fmt="%d")
    return {"model": name, "accuracy": float(accuracy_score(truth, predicted)), "macro_f1": float(f1_score(truth, predicted, labels=np.arange(len(LABELS)), average="macro", zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(truth, predicted))}

class MacroF1(tf.keras.callbacks.Callback):
    def __init__(self, validation, truth):
        self.validation, self.truth = validation, truth

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.validation, verbose=0)
        logs["val_macro_f1"] = float(f1_score(self.truth, probs.argmax(axis=1), labels=np.arange(len(LABELS)), average="macro", zero_division=0))
        print("Validation macro-F1:", round(logs["val_macro_f1"], 5), flush=True)

def main():
    import argparse
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    tf.keras.utils.set_random_seed(42)
    manifest = load_manifest()
    verification = json.loads((BASE / "data/expanded/dataset_verification.json").read_text())
    assert len(manifest) == verification["retained_images"] and len(manifest) > 100000
    assert manifest.sha256.is_unique and manifest.groupby("group_id").split.nunique().max() == 1
    frames = {split: manifest[manifest.split == split].sort_values(["label", "path"]).reset_index(drop=True) for split in ("train", "validation", "test")}
    truth = {split: frame.label.map({label: index for index, label in enumerate(LABELS)}).to_numpy(dtype=np.int32) for split, frame in frames.items()}
    counts = np.bincount(truth["train"], minlength=len(LABELS))
    assert (counts > 0).all()
    raw = np.sqrt(len(frames["train"]) / (len(LABELS) * counts))
    weights = np.minimum(raw, 3.0)
    weights /= weights.mean()
    class_weights = dict(enumerate(map(float, weights)))
    pd.DataFrame({"label": LABELS, "train_images": counts, "class_weight": weights}).to_csv(OUT / "class_weights.csv", index=False)
    print("VERIFIED DATA:", verification["split_counts"], flush=True)
    print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)
    backbone = tf.keras.applications.MobileNetV3Large(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    backbone.trainable = False
    features = {}
    for split, frame in frames.items():
        print("EXTRACTING ALL IMAGES:", split, len(frame), flush=True)
        features[split] = backbone.predict(dataset(frame), verbose=1)
        assert len(features[split]) == len(frame) and np.isfinite(features[split]).all()
    with tf.device("/CPU:0"):
        head = tf.keras.Sequential([tf.keras.Input((features["train"].shape[1],)), tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(len(LABELS), activation="softmax")])
        head.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        head.fit(features["train"], truth["train"], validation_data=(features["validation"], truth["validation"]), epochs=25, batch_size=64, class_weight=class_weights, callbacks=[MacroF1(features["validation"], truth["validation"]), tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=4, restore_best_weights=True), tf.keras.callbacks.CSVLogger(str(OUT / "frozen_training.csv"))], verbose=2)
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08, fill_mode="reflect"),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.10, 0.10),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.10, value_range=(0, 255)),
    ], name="training_only_augmentation")
    inputs = tf.keras.Input((224, 224, 3))
    model = tf.keras.Model(inputs, head(backbone(augmentation(inputs), training=False)), name="EcoSort_MobileNetV3Large")
    model.save(OUT / "frozen_baseline.keras")
    backbone.trainable = True
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= len(backbone.layers) - 60 and not isinstance(layer, tf.keras.layers.BatchNormalization)
    assert all(not layer.trainable for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
    validation = dataset(frames["validation"])
    print("FINE-TUNING: every training image is visited each epoch; no subset or steps_per_epoch cap.", flush=True)
    model.fit(dataset(frames["train"], training=True), validation_data=validation, epochs=6, class_weight=class_weights, callbacks=[MacroF1(validation, truth["validation"]), tf.keras.callbacks.ModelCheckpoint(str(OUT / "fine_tuned.keras"), monitor="val_macro_f1", mode="max", save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=3, restore_best_weights=True), tf.keras.callbacks.CSVLogger(str(OUT / "fine_training.csv"))], verbose=2)
    frozen = tf.keras.models.load_model(OUT / "frozen_baseline.keras", compile=False)
    fine = tf.keras.models.load_model(OUT / "fine_tuned.keras", compile=False)
    candidates = {"frozen_baseline": frozen, "fine_tuned": fine}
    val_probs = {name: candidate.predict(validation, verbose=0) for name, candidate in candidates.items()}
    val_f1 = {name: float(f1_score(truth["validation"], probs.argmax(axis=1), labels=np.arange(len(LABELS)), average="macro", zero_division=0)) for name, probs in val_probs.items()}
    selected = max(val_f1, key=val_f1.get)
    candidates[selected].save(OUT / "ecosort_model.keras")
    logs = np.log(np.clip(val_probs[selected], 1e-8, 1))
    def nll(temperature):
        calibrated = softmax(logs / temperature, axis=1)
        return float(-np.log(np.clip(calibrated[np.arange(len(logs)), truth["validation"]], 1e-8, 1)).mean())
    temperature = float(minimize_scalar(nll, bounds=(0.5, 5), method="bounded").x)
    calibrated = softmax(logs / temperature, axis=1)
    threshold, threshold_met = 0.85, False
    for candidate_threshold in np.arange(0.6, 0.96, 0.025):
        accepted = calibrated.max(axis=1) >= candidate_threshold
        if accepted.mean() >= 0.2 and (calibrated.argmax(axis=1)[accepted] == truth["validation"][accepted]).mean() >= 0.95:
            threshold, threshold_met = float(candidate_threshold), True
            break
    settings = {"classes": LABELS, "temperature": temperature, "review_threshold": threshold, "minimum_top_two_margin": 0.20, "manual_review_classes": [label for label, count in zip(LABELS, counts) if count < 200] + ["mixed_items", "household_problem_waste", "medicines", "light_bulbs"], "selected_by_validation": selected, "validation_macro_f1": val_f1, "threshold_target_met": threshold_met, "image_count": len(manifest), "training_images": len(frames["train"]), "backbone": "MobileNetV3Large", "training_only_augmentation": True, "full_training_epochs": len(pd.read_csv(OUT / "fine_training.csv"))}
    (OUT / "app_settings.json").write_text(json.dumps(settings, indent=2))
    (OUT / "class_names.json").write_text(json.dumps(LABELS, indent=2))
    metrics = []
    for name, candidate in candidates.items():
        probs = candidate.predict(dataset(frames["test"]), verbose=0)
        metrics.append(evaluate(name, probs, truth["test"]))
        report = frames["test"][["dataset", "label", "path"]].copy()
        report["predicted"] = [LABELS[i] for i in probs.argmax(axis=1)]
        report.to_csv(OUT / (name + "_predictions.csv"), index=False)
    pd.DataFrame(metrics).to_csv(OUT / "baseline_vs_finetuned.csv", index=False)
    completion = {"status": "completed", "verified_image_count": len(manifest), "split_counts": verification["split_counts"], "seconds": time.perf_counter() - started, "selected": selected, "test_metrics": metrics, "note": "All retained images processed; only the training partition updates weights. Validation selects checkpoints and thresholds. Test images never train the model."}
    (OUT / "training_completed.json").write_text(json.dumps(completion, indent=2))
    print(json.dumps(completion, indent=2), flush=True)

if __name__ == "__main__":
    main()
