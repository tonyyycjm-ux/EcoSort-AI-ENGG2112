"""Resume range downloads of the public Garbage265 mirror with SHA-256 checks."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import requests
import time

BASE = Path(__file__).resolve().parent / "data/garbage265_download"
BASE.mkdir(exist_ok=True, parents=True)
SOURCES = {
    "train.zip": (12204497697, "e8afee7456b9b449c22dd344839b19afe9208c69bb8ac6518d0280cc1d2f54a1"),
    "val.zip": (1348134481, "6da262bcafeddf87c878bc48e477992efc01369d7f13bc55e4e5a50c28af9c11"),
}

def download_file(url, destination, size, expected_hash):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    name = destination.name
    if destination.exists() and destination.stat().st_size == size:
        print("Already downloaded:", name, flush=True)
        return
    partial = destination.with_name(name + ".partial")
    state = destination.with_name(name + ".parts")
    state.mkdir(exist_ok=True)
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    chunk = 64 * 1024 * 1024
    tasks = [(index, start, min(start + chunk, size) - 1) for index, start in enumerate(range(0, size, chunk))]
    def fetch(task):
        index, start, end = task
        marker = state / str(index)
        if marker.exists():
            return end - start + 1
        for attempt in range(4):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                        raise ValueError("Server did not honour requested byte range")
                    written = 0
                    for block in response.iter_content(1024 * 1024):
                        assert os.pwrite(fd, block, start + written) == len(block)
                        written += len(block)
                    assert written == end - start + 1
                marker.touch()
                return written
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            for future in as_completed([pool.submit(fetch, task) for task in tasks]):
                done += future.result()
                print(name, f"{done / size:.1%}", flush=True)
    finally:
        os.close(fd)
    digest = hashlib.file_digest(partial.open("rb"), "sha256").hexdigest()
    assert digest == expected_hash, "Archive checksum mismatch"
    partial.replace(destination)
    print("SHA256 verified:", name, flush=True)

def download(name):
    size, expected_hash = SOURCES[name]
    url = "https://huggingface.co/datasets/fritz1024/garbage265-classification/resolve/main/" + name
    download_file(url, BASE / name, size, expected_hash)

if __name__ == "__main__":
    for filename in SOURCES:
        download(filename)
