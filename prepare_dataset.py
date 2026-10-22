"""Restore and hash-verify the exact previously selected 20,000-image manifest."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import ssl
import urllib.request
import zipfile
import certifi
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
CONTEXT = ssl.create_default_context(cafile=certifi.where())
JOBS = [
    ("TrashNet", "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip", DATA / "TrashNet"),
    ("RealWaste", "https://archive.ics.uci.edu/static/public/908/realwaste.zip", DATA / "RealWaste"),
    ("household", "https://www.kaggle.com/api/v1/datasets/download/mostafaabla/garbage-classification", DATA / "EcoSort_dataset_v3/raw/household"),
    ("ewaste", "https://www.kaggle.com/api/v1/datasets/download/akshat103/e-waste-image-dataset", DATA / "EcoSort_dataset_v3/raw/ewaste"),
    ("phenom", "https://www.kaggle.com/api/v1/datasets/download/phenomsg/waste-classification", DATA / "EcoSort_dataset_v3/raw/phenom"),
]

def restore(job):
    name, url, destination = job
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".verified_extraction"
    if marker.exists():
        return name
    archive = DATA / (name + ".zip")
    if name == "TrashNet" and not archive.exists():
        existing = Path.home() / "Desktop/ENGG 2112/project/TrashNet/dataset-resized.zip"
        if existing.exists():
            shutil.copy2(existing, archive)
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        print("Downloading", name, flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "EcoSort coursework"})
        with urllib.request.urlopen(request, context=CONTEXT, timeout=180) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        partial.replace(archive)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive member")
        source.extractall(destination)
    marker.touch()
    print("Extracted", name, flush=True)
    return name

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(restore, JOBS))
    manifest = pd.read_csv(BASE / "saved_results/split_manifest.csv")
    assert len(manifest) == 20000 and manifest.group_id.is_unique
    manifest["original_path"] = manifest.path
    manifest["path"] = manifest.path.map(lambda path: str(DATA / Path(path).relative_to("/content")))
    def verify(row):
        path = Path(row.path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
            raise ValueError("Source image changed: " + str(path))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify, manifest.itertuples()))
    assert manifest.groupby("group_id").split.nunique().max() == 1
    manifest.to_csv(DATA / "split_manifest_local.csv", index=False)
    report = {"total_verified_images": len(manifest), "splits": manifest.groupby("split").size().to_dict(), "class_counts": manifest.groupby("label").size().to_dict(), "sha256_verified": True}
    (DATA / "dataset_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
