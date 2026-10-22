"""Collect a new whole-photo panel without consulting any model predictions."""
import hashlib
import json
from pathlib import Path
import time
import requests
from PIL import Image,ImageDraw,ImageOps
from collect_external import clean

BASE = Path(__file__).resolve().parent
OUT = BASE / "final_test"
QUERIES = [("plastic","PET plastic bottle"),("glass","empty glass jar"),("ceramics","ceramic bowl"),("metal","tin can"),("cardboard","corrugated cardboard box"),("paper","paper sheets"),("battery","alkaline battery"),("ewaste","computer mouse"),("food_organics","apple core"),("textile_trash","cotton shirt"),("glass","empty jam jar"),("metal","single aluminium can"),("textile_trash","shirt hanger")]
QUERIES += [("glass",'intitle:"glass jar"'),("metal",'intitle:"beverage can"'),("food_organics",'intitle:"banana peel"')]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "test_manifest.json").exists(), "Do not change a frozen panel."
    seen = {row["title"] for row in json.loads((BASE / "external_test/candidates.json").read_text())}
    session = requests.Session()
    session.headers["User-Agent"] = "EcoSortCourseworkEvaluation/1.0"
    entries = json.loads((OUT/"candidates.json").read_text()) if (OUT/"candidates.json").exists() else []
    for index,(label,query) in enumerate(QUERIES):
        if any(row["group"] == index for row in entries):
            continue
        params={"action":"query","generator":"search","gsrsearch":query+" filetype:bitmap","gsrnamespace":6,"gsrlimit":6,"prop":"imageinfo","iiprop":"url|extmetadata","iiurlwidth":640,"format":"json"}
        response=session.get("https://commons.wikimedia.org/w/api.php",params=params,timeout=30)
        response.raise_for_status()
        for rank,page in enumerate(sorted(response.json().get("query",{}).get("pages",{}).values(),key=lambda r:r["index"])):
            if page["title"] in seen or any(t in page["title"].lower() for t in (".svg","diagram","icon","illustration","logo",".gif")):
                continue
            info=page["imageinfo"][0]
            meta=info["extmetadata"]
            path=OUT/f"candidate_{index}_{rank}.jpg"
            try:
                if not path.exists():
                    data=session.get(info.get("thumburl",info["url"]),timeout=20)
                    data.raise_for_status()
                    path.write_bytes(data.content)
                with Image.open(path) as image:
                    image.verify()
                entries.append({"id":f"{index}:{rank}","group":index,"rank":rank,"expected_label":label,"title":page["title"],"path":str(path),"source_url":info["descriptionurl"],"author":clean(meta.get("Artist",{}).get("value","")),"licence":clean(meta.get("LicenseShortName",{}).get("value","")),"description":clean(meta.get("ImageDescription",{}).get("value","")),"photo_date":clean(meta.get("DateTimeOriginal",{}).get("value","")),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
                (OUT/"candidates.json").write_text(json.dumps(entries,indent=2))
                time.sleep(.6)
            except Exception as error:
                print("Unavailable candidate:",page["title"],str(error)[:80],flush=True)
        print("Collected whole-photo candidates:",label,flush=True)
    canvas=Image.new("RGB",(1200,220*len(QUERIES)),"white")
    draw=ImageDraw.Draw(canvas)
    for row in entries:
        x,y=row["rank"]*200,row["group"]*220
        with Image.open(row["path"]) as image:
            canvas.paste(ImageOps.contain(image.convert("RGB"),(195,185)),(x,y+25))
        draw.text((x+2,y+3),row["id"]+" "+row["expected_label"],fill="black")
    canvas.save(OUT/"candidate_review.jpg")
