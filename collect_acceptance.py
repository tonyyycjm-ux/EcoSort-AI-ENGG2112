"""Collect candidates in fixed search order, without calling a prediction model."""
import hashlib
import json
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps
from collect_external import clean

BASE = Path(__file__).resolve().parent
OUT = BASE/"acceptance_web"
QUERIES = [
    ("plastic", "plastic water bottles"), ("plastic", "plastic food container"),
    ("glass", "glass bottles"), ("glass", "preserving glass jars"),
    ("metal", "aluminium beverage cans"), ("metal", "steel food cans"),
    ("cardboard", "cardboard shipping boxes"), ("paper", "newspaper stack"),
    ("battery", "AA batteries"), ("ewaste", "computer keyboards"),
    ("textile_trash", "cotton t shirts"), ("ceramics", "porcelain mugs"),
    ("food_organics", "banana peels"), ("vegetation", "garden grass clippings"),
    ("wood", "wooden spoons"), ("plastic", "plastic shopping bags"),
    ("trash", "paper towels"), ("light_bulbs", "compact fluorescent lamps"),
    ("medicines", "medicine tablets blister"), ("household_problem_waste", "household paint cans"),
    ("battery", 'intitle:"battery" alkaline'),
    ("metal", 'intitle:"can" beverage'),
    ("metal", 'intitle:"tin can"'),
    ("food_organics", 'intitle:"apple core"'),
    ("vegetation", 'intitle:"grass clippings"'),
    ("household_problem_waste", 'intitle:"paint cans"'),
    ("household_problem_waste", 'intitle:"paint can"'),
    ("vegetation", 'intitle:"leaf litter"'),
    ("food_organics", 'intitle:"banana skin"'),
    ("metal", 'intitle:"canned food"'),
    ("food_organics", 'intitle:"orange peel"'),
    ("food_organics", 'intitle:"food waste"'),
    ("vegetation", 'intitle:"leaf pile"'),
    ("food_organics", '"food scraps"'),
    ("vegetation", '"fallen leaves" -painting'),
    ("food_organics", "File:Banana peel 2.jpg"),
    ("vegetation", "File:Fallen Leaves.jpg"),
    ("food_organics", "File:Banana peel on seat.JPG"),
]


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/"test_manifest.json").exists(), "Acceptance images are frozen."
    target = OUT/"candidates.json"
    rows = json.loads(target.read_text()) if target.exists() else []
    previous = set()
    for folder in ("external_test", "final_test"):
        previous.update(r["title"] for r in json.loads((BASE/folder/"candidates.json").read_text()))
    session = requests.Session()
    session.headers["User-Agent"] = "EcoSortCourseworkEvaluation/2.0 (educational photo evaluation)"
    for slot, (label, query) in enumerate(QUERIES):
        if any(r["slot"] == slot for r in rows):
            continue
        params = dict(action="query", generator="search", gsrsearch=query+" filetype:bitmap",
                      gsrnamespace=6, gsrlimit=6, prop="imageinfo", iiprop="url|extmetadata",
                      iiurlwidth=640, format="json")
        if query.startswith("File:"):
            params = dict(action="query", titles=query, prop="imageinfo", iiprop="url|extmetadata",
                          iiurlwidth=640, format="json")
        response = session.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=40)
        response.raise_for_status()
        pages = sorted(response.json().get("query", {}).get("pages", {}).values(), key=lambda r: r.get("index", 0))
        for rank, page in enumerate(pages):
            title = page["title"]
            if title in previous or any(x in title.lower() for x in (".svg", ".gif", "diagram", "icon", "logo", "illustration")):
                continue
            info = page["imageinfo"][0]
            meta = info["extmetadata"]
            path = OUT/f"photo_{slot:02}_{rank}.jpg"
            try:
                if not path.exists():
                    data = session.get(info.get("thumburl", info["url"]), timeout=30)
                    data.raise_for_status()
                    path.write_bytes(data.content)
                with Image.open(path) as image:
                    image.verify()
                rows.append(dict(id=f"{slot}:{rank}", slot=slot, rank=rank, expected_label=label,
                                 query=query, title=title, path=str(path), source_url=info["descriptionurl"],
                                 author=clean(meta.get("Artist", {}).get("value", "")),
                                 licence=clean(meta.get("LicenseShortName", {}).get("value", "")),
                                 description=clean(meta.get("ImageDescription", {}).get("value", "")),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                target.write_text(json.dumps(rows, indent=2))
                time.sleep(1)
            except (requests.RequestException, OSError) as error:
                print("Unavailable", title, str(error)[:100], flush=True)
        print("Collected slot", slot, query, flush=True)
    for start in range(0, len(QUERIES), 5):
        canvas = Image.new("RGB", (1440, 1200), "white")
        draw = ImageDraw.Draw(canvas)
        for row in rows:
            if start <= row["slot"] < start+5:
                x, y = row["rank"]*240, (row["slot"]-start)*240
                with Image.open(row["path"]) as image:
                    canvas.paste(ImageOps.contain(image.convert("RGB"), (230, 205)), (x, y+30))
                draw.text((x+3, y+3), row["id"]+" "+row["expected_label"], fill="black")
        canvas.save(OUT/f"review_{start:02}.jpg")


if __name__ == "__main__":
    main()
