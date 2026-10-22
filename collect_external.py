"""Collect an external test panel before running any model predictions."""
import hashlib
import html
import json
from pathlib import Path
import re
import time
import requests
from PIL import Image, ImageDraw, ImageOps

BASE = Path(__file__).resolve().parent
OUT = BASE / "external_test"
OUT.mkdir(exist_ok=True)
QUERIES = [
    ("plastic", "plastic water bottle filetype:bitmap"),
    ("glass", "empty glass bottle filetype:bitmap"),
    ("ceramics", "ceramic mug filetype:bitmap"),
    ("metal", "aluminium beverage can filetype:bitmap"),
    ("cardboard", "cardboard box filetype:bitmap"),
    ("paper", "stack of newspapers filetype:bitmap"),
    ("battery", "AA batteries filetype:bitmap"),
    ("ewaste", "computer keyboard filetype:bitmap"),
    ("food_organics", "banana peel filetype:bitmap"),
    ("textile_trash", "cotton t-shirt filetype:bitmap"),
]

def clean(value):
    return html.unescape(re.sub("<[^>]+>", "", value))

if __name__ == "__main__":
    session = requests.Session()
    session.headers["User-Agent"] = "EcoSortCourseworkImageEvaluation/1.0"
    entries = []
    for index, (label, query) in enumerate(QUERIES):
        params = {"action":"query", "generator":"search", "gsrsearch":query, "gsrnamespace":6, "gsrlimit":3, "prop":"imageinfo", "iiprop":"url|extmetadata", "iiurlwidth":1024, "format":"json"}
        response = session.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=45)
        response.raise_for_status()
        pages = sorted(response.json()["query"]["pages"].values(), key=lambda row: row["index"])
        for rank, page in enumerate(pages):
            info = page["imageinfo"][0]
            metadata = info["extmetadata"]
            if any(term in page["title"].lower() for term in (".svg", "diagram", "illustration", ".gif")):
                continue
            url = info.get("thumburl", info["url"])
            path = OUT / f"candidate_{index:02d}_{rank}.jpg"
            if not path.exists():
                data = session.get(url, timeout=45)
                if data.status_code == 429:
                    print("Rate limited candidate, not evaluated:", page["title"], flush=True)
                    time.sleep(3)
                    continue
                data.raise_for_status()
                path.write_bytes(data.content)
            with Image.open(path) as image:
                image.verify()
            entry = {"test_index":index, "candidate_rank":rank, "expected_label":label, "query":query, "title":page["title"], "path":str(path), "source":info["descriptionurl"], "image_url":url, "author":clean(metadata.get("Artist",{}).get("value","")), "licence":clean(metadata.get("LicenseShortName",{}).get("value","")), "description":clean(metadata.get("ImageDescription",{}).get("value","")), "photo_date":clean(metadata.get("DateTimeOriginal",{}).get("value","")), "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            entries.append(entry)
            (OUT / "candidates.json").write_text(json.dumps(entries, indent=2))
            time.sleep(1)
        print("Collected candidate group:", label, flush=True)
        time.sleep(0.5)
    (OUT / "candidates.json").write_text(json.dumps(entries, indent=2))
    canvas = Image.new("RGB", (900, 200 * len(QUERIES)), "white")
    draw = ImageDraw.Draw(canvas)
    for entry in entries:
        x, y = entry["candidate_rank"] * 300, entry["test_index"] * 200
        with Image.open(entry["path"]) as source:
            canvas.paste(ImageOps.contain(source.convert("RGB"), (290, 170)), (x, y + 25))
        draw.text((x + 2, y + 2), f"{entry['test_index']}:{entry['candidate_rank']} {entry['expected_label']}", fill="black")
    canvas.save(OUT / "candidate_review.jpg")
