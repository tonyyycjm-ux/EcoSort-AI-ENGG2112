"""Preserve the Colab export and create explicitly labelled local copies."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

BASE = Path(__file__).resolve().parent
assert not (BASE / "ECOSORT_Legacy_12class_Run.ipynb").exists(), "Legacy migration is complete; use the expanded training notebook."
SOURCE = Path.home() / "Downloads" / "ECOSORT_AI (1).ipynb"
ORIGINAL = BASE / "ECOSORT_Colab_Original.ipynb"
shutil.copy2(SOURCE, ORIGINAL)
assert SOURCE.read_bytes() == ORIGINAL.read_bytes()
original = json.loads(ORIGINAL.read_text())
assert len(original["cells"]) == 58
with zipfile.ZipFile(Path.home() / "Downloads/EcoSort_balanced_12class_results.zip") as archive:
    destination = BASE / "saved_results"
    for member in archive.infolist():
        assert (destination / member.filename).resolve().is_relative_to(destination.resolve())
    archive.extractall(destination)

def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True), "execution_count": None, "outputs": []}

kernel = {"display_name": "EcoSort Local (Python 3.12 / Metal)", "language": "python", "name": "ecosort-local"}
metadata = {"kernelspec": kernel, "language_info": {"name": "python", "version": "3.12"}}
local = copy.deepcopy(original)
local["metadata"] = copy.deepcopy(metadata)
changes = []
for i, cell in enumerate(local["cells"]):
    if cell["cell_type"] != "code":
        continue
    before = "".join(cell["source"])
    # Replace complete Path expressions, keeping all other algorithm code intact.
    import re
    after = re.sub(r'Path\("/content/([^"\n]+)"\)', r'(LOCAL_CONTENT / "\1")', before)
    after = after.replace('!pip install -q codecarbon', '# Dependencies are installed in the project virtual environment.')
    after = after.replace('from google.colab import files', 'from IPython.display import FileLink')
    after = after.replace('files.download(bundle)', 'display(FileLink(bundle))')
    after = after.replace('files.download(archive3)', 'display(FileLink(archive3))')
    after = after.replace('share=True, inline=True, debug=False', 'share=False, server_name="127.0.0.1", inline=True, debug=False')
    after = after.replace('# 从 1e-5 提升至 3e-5，加速 CPU 下的收敛', '# Increased from 1e-5 to 3e-5 for faster CPU convergence.')
    cell["source"] = after.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None
    cell["metadata"] = {}
    if before != after:
        changes.append({"original_cell_index": i, "original_sha256": hashlib.sha256(before.encode()).hexdigest(), "local_sha256": hashlib.sha256(after.encode()).hexdigest()})

startup = '''from pathlib import Path
import os
import ssl
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
PROJECT = Path.cwd()
assert (PROJECT / "ECOSORT_Colab_Original.ipynb").exists(), "Open this notebook from the EcoSort-Local project folder."
LOCAL_CONTENT = PROJECT / "data"
LOCAL_CONTENT.mkdir(exist_ok=True)
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
'''
local["cells"][:0] = [markdown('''# EcoSort AI - Local Training Copy

The original 58 cells remain in their original order below. All original Markdown explanations are retained. Local-only changes: project-relative data paths, preinstalled dependencies, local file links instead of Colab downloads, localhost-only Gradio, and one translated comment. Old outputs have been cleared to avoid presenting cloud results as local results. The byte-for-byte Colab export is ECOSORT_Colab_Original.ipynb.

Run this notebook from top to bottom to download the source datasets and retrain. This is separate from ECOSORT_Local_Run.ipynb, which loads the already-trained 12-class model without retraining. The full dataset is not included in the model evidence archive. Existing CodeCarbon estimates are not calibrated hardware energy measurements on this Mac.

## Local environment and storage

Use the EcoSort Local kernel. Files stay inside this project instead of the Colab /content directory. The cell below verifies the environment and enables trusted HTTPS certificate validation.
'''), code(startup)]
(BASE / "ECOSORT_Local_Training.ipynb").write_text(json.dumps(local, indent=1, ensure_ascii=False))
(BASE / "migration_changes.json").write_text(json.dumps(changes, indent=2))

load = '''from pathlib import Path
import json
import os
import time
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image, ImageOps
from scipy.special import softmax

PROJECT = Path.cwd()
RESULTS = PROJECT / "saved_results"
assert RESULTS.is_dir(), "Open this notebook from the EcoSort-Local project folder."
settings = json.loads((RESULTS / "app_settings.json").read_text())
LABELS_V3 = json.loads((RESULTS / "class_names.json").read_text())
assert settings["classes"] == LABELS_V3 and len(LABELS_V3) == 12
temperature3 = float(settings["temperature"])
review_threshold3 = float(settings["review_threshold"])
weighted_model3 = tf.keras.models.load_model(RESULTS / "ecosort_weighted_12class.keras", compile=False)
assert weighted_model3.output_shape[-1] == 12
print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print("Loaded unchanged saved 12-class model; no retraining.")
'''
original_app = "".join(original["cells"][52]["source"])
classifier = original_app.split('\nsample3 = ')[0]
assert 'def classify_balanced_waste(image):' in classifier
sample = '''import urllib.request
import ssl
import certifi
import zipfile
import hashlib

SAMPLES = PROJECT / "samples"
SAMPLES.mkdir(exist_ok=True)
SAMPLE = SAMPLES / "cardboard1.jpg"
if not SAMPLE.exists():
    context = ssl.create_default_context(cafile=certifi.where())
    url = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
    with urllib.request.urlopen(url, context=context, timeout=120) as response:
        archive_data = response.read()
    import io
    with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
        SAMPLE.write_bytes(archive.read("dataset-resized/cardboard/cardboard1.jpg"))

with Image.open(SAMPLE) as sample:
    scores_check3, message_check3 = classify_balanced_waste(sample)
assert len(scores_check3) == 12 and np.isclose(sum(scores_check3.values()), 1.0)
assert all(np.isfinite(list(scores_check3.values())))
try:
    classify_balanced_waste(None)
except gr.Error:
    pass
else:
    raise AssertionError("Missing-image validation must reject empty input.")
durations = []
with Image.open(SAMPLE) as sample:
    for _ in range(5):
        started = time.perf_counter()
        classify_balanced_waste(sample)
        durations.append((time.perf_counter() - started) * 1000)
verification = {
    "tensorflow": tf.__version__,
    "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
    "model_sha256": hashlib.sha256((RESULTS / "ecosort_weighted_12class.keras").read_bytes()).hexdigest(),
    "sample": str(SAMPLE), "decision": message_check3,
    "median_warm_prediction_ms": float(np.median(durations)),
    "scores": scores_check3,
    "scope": "Local inference smoke test only; not a new accuracy evaluation or full retraining.",
}
(PROJECT / "local_verification.json").write_text(json.dumps(verification, indent=2))
print(json.dumps(verification, indent=2))
display(sample.copy())
'''
launch = '''if "demo3" in globals():
    demo3.close()
demo3 = gr.Interface(
    fn=classify_balanced_waste,
    inputs=gr.Image(sources=["upload", "webcam"], type="pil", label="Waste item"),
    outputs=[gr.Label(num_top_classes=12, label="Category scores"), gr.Textbox(label="Decision", interactive=False)],
    examples=[[str(SAMPLE)]],
    title="EcoSort AI", flagging_mode="never",
)
demo3.launch(share=False, server_name="127.0.0.1", server_port=7860, inline=False, inbrowser=False, prevent_thread_lock=True)
'''
quick = {"nbformat": 4, "nbformat_minor": 5, "metadata": metadata, "cells": [
    markdown('''# EcoSort AI - Local Run

This notebook runs the original saved 12-class model on this Mac without retraining or changing the classifier. Use the EcoSort Local kernel. The complete 58-cell training notebook and unchanged Colab export are separate files in this folder.

## 1. Load the saved model and original calibration

Load the exact model weights, category order, temperature and review threshold from the downloaded evidence archive. Print available GPU devices. No cloud connection is needed for predictions.
'''), code(load),
    markdown('''## 2. Original prediction function

The following definition is copied verbatim from Colab cell 52. It corrects EXIF orientation, converts to RGB, resizes to 224 by 224, applies the original model and temperature scaling, and requests manual review for low-confidence or closely competing predictions.
'''), code(classifier),
    markdown('''## 3. Run and record a local smoke test

Download one public TrashNet cardboard sample on first run, verify the 12 scores sum to one, check missing-image validation, and time five warm predictions. This confirms local operation, not model accuracy on a new dataset. No webcam permissions are requested by this test.
'''), code(sample),
    markdown('''## 4. Open the local application

Start the original upload/webcam interface on http://127.0.0.1:7860. Only this Mac can access it; no Gradio public tunnel is created. Keep this notebook kernel running while using the application. Close any existing standalone EcoSort application before running this cell, because both use port 7860.
'''), code(launch),
]}
for notebook in [quick, local]:
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"ecosort-{index}"
(BASE / "ECOSORT_Local_Run.ipynb").write_text(json.dumps(quick, indent=1))
(BASE / "ECOSORT_Local_Training.ipynb").write_text(json.dumps(local, indent=1, ensure_ascii=False))
assert "".join(quick["cells"][3]["source"]) == original_app.split('\nsample3 = ')[0]
assert len(local["cells"]) == len(original["cells"]) + 2
for old, new in zip(original["cells"], local["cells"][2:]):
    if old["cell_type"] == "markdown":
        assert old["source"] == new["source"]
print("Original preserved byte-for-byte; all 58 cells and Markdown retained.")
