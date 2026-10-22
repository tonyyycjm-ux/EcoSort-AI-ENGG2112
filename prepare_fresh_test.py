"""Prepare a new, prediction-blind TACO object-crop panel with source labels."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import random
from pathlib import Path
import time
import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from prepare_expanded import audit

BASE = Path(__file__).resolve().parent
OUT = BASE / "fresh_test"
MAPPING = {4:"plastic", 5:"plastic", 21:"plastic", 24:"plastic", 43:"plastic", 44:"plastic", 45:"plastic", 47:"plastic", 6:"glass", 9:"glass", 23:"glass", 26:"glass", 0:"metal", 8:"metal", 10:"metal", 12:"metal", 28:"metal", 52:"metal", 15:"cardboard", 17:"cardboard", 19:"cardboard", 30:"paper", 32:"paper", 33:"paper", 34:"paper", 25:"food_organics", 31:"trash", 59:"trash"}

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "test_manifest.json").exists(), "Do not change a frozen panel."
    session = requests.Session()
    session.headers["User-Agent"] = "EcoSortCourseworkEvaluation/1.0"
    metadata = OUT / "annotations.json"
    if not metadata.exists():
        response = session.get("https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json", timeout=45)
        response.raise_for_status()
        metadata.write_bytes(response.content)
    data = json.loads(metadata.read_text())
    images = {row["id"]:row for row in data["images"]}
    categories = {row["id"]:row["name"] for row in data["categories"]}
    pool = pd.read_csv(BASE / "data/expanded/raw_audit.csv", dtype={"phash":"string"})
    pool = pool[pool.status == "valid"]
    sha, pixels = set(pool.sha256), set(pool.pixel_hash)
    hashes = [int(value) for value in pool.phash]
    candidates = [row for row in data["annotations"] if row["category_id"] in MAPPING and not row.get("iscrowd")]
    random.Random(20261022).shuffle(candidates)
    candidates = [row for row in candidates if images[row["image_id"]].get("flickr_640_url") and min(row["bbox"][2:])*640/max(images[row["image_id"]]["width"],images[row["image_id"]]["height"]) >= 60 and row["bbox"][2]*row["bbox"][3]/(images[row["image_id"]]["width"]*images[row["image_id"]]["height"]) >= 0.04]
    def fetch(image_id):
        path = OUT / f"source_{image_id}.jpg"
        if not path.exists():
            try:
                response = requests.get(images[image_id]["flickr_640_url"], timeout=(5,10))
                response.raise_for_status()
                path.write_bytes(response.content)
            except requests.RequestException:
                return
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(fetch, sorted({row["image_id"] for row in candidates})))
    chosen, used = [], set()
    counts = {label:0 for label in sorted(set(MAPPING.values()))}
    failures = []
    for annotation in candidates:
        label = MAPPING[annotation["category_id"]]
        info = images[annotation["image_id"]]
        if counts[label] >= 10 or info["id"] in used:
            continue
        x,y,w,h = annotation["bbox"]
        if min(w,h) < 100 or w*h/(info["width"]*info["height"]) < 0.04:
            continue
        path = OUT / f"source_{info['id']}.jpg"
        url = info.get("flickr_640_url") or info["flickr_url"]
        try:
            if not path.exists():
                raise ValueError("Thumbnail unavailable at collection time")
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                sx,sy = image.width/info["width"],image.height/info["height"]
                assert abs(sx/sy-1) < 0.03, "Annotation/image orientation mismatch"
                margin = 0.08
                box = (max(0,int((x-w*margin)*sx)),max(0,int((y-h*margin)*sy)),min(image.width,int((x+w*(1+margin))*sx)),min(image.height,int((y+h*(1+margin))*sy)))
                crop = image.crop(box)
                assert min(crop.size) >= 64, "Object too small after download"
                target = OUT / f"crop_{annotation['id']}.jpg"
                crop.save(target, quality=95)
            checked = audit({"path":str(target)})
            full = audit({"path":str(path)})
            assert checked["status"] == "valid" and full["status"] == "valid"
            distances = [min((int(row["phash"]) ^ value).bit_count() for value in hashes) for row in (checked,full)]
            if any(row["sha256"] in sha or row["pixel_hash"] in pixels for row in (checked,full)) or min(distances) <= 2:
                failures.append({"annotation":annotation["id"],"reason":"Duplicate or near match"})
                continue
            chosen.append({"id":annotation["id"],"image_id":info["id"],"expected_label":label,"source_category":categories[annotation["category_id"]],"path":str(target),"source_image":str(path),"source_url":info["flickr_url"],"annotation_source":"https://github.com/pedropro/TACO/blob/master/data/annotations.json","bbox":annotation["bbox"],"crop_margin":margin,"sha256":checked["sha256"],"minimum_crop_phash_distance":distances[0],"minimum_source_phash_distance":distances[1],"licence":info.get("license")})
            used.add(info["id"])
            counts[label] += 1
            (OUT / "candidates.json").write_text(json.dumps(chosen, indent=2))
            print("Collected",label,counts[label],"annotation",annotation["id"],flush=True)
            time.sleep(0.2)
        except Exception as error:
            failures.append({"annotation":annotation["id"],"reason":str(error)})
        if all(value >= 10 for value in counts.values()):
            break
    (OUT / "collection_failures.json").write_text(json.dumps(failures,indent=2))
    canvas = Image.new("RGB", (1400, 220*((len(chosen)+6)//7)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(chosen):
        x,y = index%7*200,index//7*220
        with Image.open(row["path"]) as image:
            canvas.paste(ImageOps.contain(image,(195,180)),(x,y+35))
        draw.multiline_text((x+2,y+2),f"{row['id']} {row['expected_label']}\n{row['source_category']}",fill="black")
    canvas.save(OUT / "candidate_review.jpg")
    print("PREDICTION-BLIND candidates:", counts, flush=True)

if __name__ == "__main__":
    main()
