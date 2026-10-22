"""Restore source-attributed acceptance photos, rejecting any changed bytes."""
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import unquote, urlparse

import requests

BASE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((BASE/"acceptance_web/test_manifest.json").read_text())
    session = requests.Session()
    session.headers["User-Agent"] = "EcoSortCourseworkReproduction/1.0"
    for item in manifest["images"]:
        destination = BASE/"acceptance_web"/Path(item["path"]).name
        if destination.exists():
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == item["sha256"]
            continue
        parsed = urlparse(item["source_url"])
        assert parsed.scheme == "https" and parsed.netloc == "commons.wikimedia.org"
        title = unquote(parsed.path.removeprefix("/wiki/"))
        assert title.startswith("File:")
        response = session.get("https://commons.wikimedia.org/w/api.php", params=dict(
            action="query", titles=title, prop="imageinfo", iiprop="url", iiurlwidth=640, format="json"), timeout=40)
        response.raise_for_status()
        page = next(iter(response.json()["query"]["pages"].values()))
        info = page["imageinfo"][0]
        url = info.get("thumburl", info["url"])
        assert urlparse(url).netloc == "upload.wikimedia.org"
        response = session.get(url, timeout=40)
        response.raise_for_status()
        assert hashlib.sha256(response.content).hexdigest() == item["sha256"], "Source changed; do not substitute a photo."
        destination.write_bytes(response.content)
        print("Restored", item["id"], flush=True)
        time.sleep(2)
    print("All 60 acceptance photo checksums verified.")


if __name__ == "__main__":
    main()
