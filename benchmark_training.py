"""Measure the proposed real-image pipeline without saving trained weights."""
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from train_full import BASE, LABELS, dataset, tf

tf.keras.utils.set_random_seed(42)
manifest = pd.read_csv(BASE / "saved_results/split_manifest.csv")
manifest["path"] = manifest.path.map(lambda path: str(BASE / "data" / Path(path).relative_to("/content")))
available = manifest[(manifest.split == "train") & manifest.path.map(lambda path: Path(path).is_file())]
frame = available.sample(n=min(2048, len(available)), random_state=42).reset_index(drop=True)
assert len(frame) >= 512
print("Available real training images:", len(available), "Benchmark images:", len(frame), flush=True)
backbone = tf.keras.applications.MobileNetV3Large(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
backbone.trainable = False
data = dataset(frame).repeat()
backbone.predict(data.take(3), verbose=0)
start = time.perf_counter()
backbone.predict(data.take(24), verbose=0)
feature_seconds_per_batch = (time.perf_counter() - start) / 24
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.08, fill_mode="reflect"),
    tf.keras.layers.RandomZoom(0.15), tf.keras.layers.RandomTranslation(0.10, 0.10),
    tf.keras.layers.RandomContrast(0.15), tf.keras.layers.RandomBrightness(0.10, value_range=(0, 255)),
])
inputs = tf.keras.Input((224, 224, 3))
head = tf.keras.Sequential([tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(len(LABELS), activation="softmax")])
model = tf.keras.Model(inputs, head(backbone(augmentation(inputs), training=False)))
backbone.trainable = True
for index, layer in enumerate(backbone.layers):
    layer.trainable = index >= len(backbone.layers) - 60 and not isinstance(layer, tf.keras.layers.BatchNormalization)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
warm_start = time.perf_counter()
model.fit(data, steps_per_epoch=4, epochs=1, verbose=0)
warmup_seconds = time.perf_counter() - warm_start
start = time.perf_counter()
model.fit(data, steps_per_epoch=24, epochs=1, verbose=0)
training_seconds_per_batch = (time.perf_counter() - start) / 24
with tf.device("/CPU:0"):
    fake_features = np.random.default_rng(42).normal(size=(14001, 960)).astype("float32")
    fake_labels = np.random.default_rng(42).integers(len(LABELS), size=14001)
    small_head = tf.keras.Sequential([tf.keras.Input((960,)), tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(len(LABELS), activation="softmax")])
    small_head.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    start = time.perf_counter()
    small_head.fit(fake_features, fake_labels, batch_size=64, epochs=2, verbose=0)
    head_seconds_per_epoch = (time.perf_counter() - start) / 2
epoch_seconds = 219 * training_seconds_per_batch + 2 * 47 * feature_seconds_per_batch
fixed_seconds = 313 * feature_seconds_per_batch + 25 * head_seconds_per_epoch + warmup_seconds + 240 * feature_seconds_per_batch
result = {
    "hardware": "Apple M4 Max, 36 GB", "gpu": [str(x) for x in tf.config.list_physical_devices("GPU")],
    "architecture": "MobileNetV3Large, 224x224, batch 64, last 60 layers except BatchNorm fine-tuned",
    "measured_real_image_batches": 24, "benchmark_training_images": len(frame),
    "feature_seconds_per_batch": feature_seconds_per_batch,
    "training_seconds_per_batch": training_seconds_per_batch,
    "graph_warmup_seconds": warmup_seconds,
    "cached_head_seconds_per_epoch_synthetic_features": head_seconds_per_epoch,
    "estimated_finetune_epoch_seconds_with_validation": epoch_seconds,
    "estimated_full_pipeline_seconds_3_finetune_epochs": fixed_seconds + 3 * epoch_seconds,
    "estimated_full_pipeline_seconds_6_finetune_epochs": fixed_seconds + 6 * epoch_seconds,
    "exclusions": "Dataset download and external 10-image test, UI work, camera work; estimate, not completed full training. Head timing uses synthetic features; convolution timings use real training images.",
}
(BASE / "training_time_benchmark.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
