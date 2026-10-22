"""Restore only required RealWaste originals directly from the authors' repository."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import threading
import time
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
local = threading.local()

def fetch(row):
    path = BASE / "data" / Path(row.path).relative_to("/content")
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256:
        return
    url = "https://raw.githubusercontent.com/sam-single/realwaste/main/" + row.path.split("realwaste-main/", 1)[1]
    if not hasattr(local, "session"):
        local.session = requests.Session()
    for attempt in range(4):
        try:
            response = local.session.get(url, timeout=45)
            response.raise_for_status()
            assert hashlib.sha256(response.content).hexdigest() == row.sha256
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(response.content)
            temporary.replace(path)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

if __name__ == "__main__":
    manifest = pd.read_csv(BASE / "saved_results/split_manifest.csv")
    rows = list(manifest[manifest.dataset == "RealWaste"].itertuples())
    with ThreadPoolExecutor(max_workers=20) as pool:
        for index, _ in enumerate(pool.map(fetch, rows), 1):
            if index % 500 == 0:
                print("RealWaste verified:", index, "/", len(rows), flush=True)
    print("Required RealWaste originals verified:", len(rows), flush=True)
